import os
import re
import time
import json
import yaml
import torch
import pandas as pd
import numpy as np
import evaluate
import math
# from torch.utils.tensorboard import SummaryWriter
from datasets import load_metric
from datasets import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification
from transformers import TrainingArguments
from transformers import Trainer
from transformers import DataCollatorForTokenClassification
# costumize trainer
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
from transformers.modeling_utils import unwrap_model
from transformers.trainer_utils import speed_metrics
from transformers.debug_utils import DebugOption
from transformers.utils import (
                                is_sagemaker_mp_enabled, 
                                is_apex_available,
                                is_torch_tpu_available,
)
from typing import Union
from torch import nn
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
if is_apex_available():
    from apex import amp
if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm

# 自定义的Trainer达到上文不计入loss的目的
class CustomerTrainer(Trainer):

    # def __init__(
    #     self,
    #     model: Union[PreTrainedModel, nn.Module] = None,
    #     args: TrainingArguments = None,
    #     data_collator: Optional[DataCollator] = None,
    #     train_dataset: Optional[Dataset] = None,
    #     eval_dataset: Optional[Dataset] = None,
    #     tokenizer: Optional[PreTrainedTokenizerBase] = None,
    #     model_init: Callable[[], PreTrainedModel] = None,
    #     compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
    #     callbacks: Optional[List[TrainerCallback]] = None,
    #     optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
    #     preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
    # ):
    #     super().

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )
        self.log(output.metrics)
        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)
        self._memory_tracker.stop_and_update_metrics(output.metrics)
        return output.metrics

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        ###############
        # 这个label_smoother一般是在使用crossEntropy的时候才使用
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        ################

        # import pdb;pdb.set_trace()
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss


class MyTrainer:

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config["device"])
        self.metric = evaluate.load("seqeval")

        self.model_path = None
        self.model = None
        self.tokenizer = None
        self.data_collator = None
        self._load_model(None)
        # self.label_encoding_dict = {'B-LOC': 0, 'B-MISC': 1, 'B-ORG': 2, 'I-LOC': 3, 'I-MISC': 4, 'I-ORG': 5, 'I-PER': 6, 'O': 7}
        self.label_encoding_dict = self.model.config.label2id
        # self.label_list = ["B-LOC", "B-MISC", "B-ORG", "I-LOC", "I-MISC", "I-ORG", "I-PER", "O"]
        self.label_list = [key for key in self.label_encoding_dict.keys()]
        self.config["HAVE_PRE"] = self._check_have_pre()

        self.valid_data = None
        self.train_data = None
        self._init_data()
        self.valid_data = self._get_labels(self.valid_data)
        self.train_data = self._get_labels(self.train_data)

        self.train_dataset = None
        self.valid_dataset = None 
        self.train_tokenized_dataset = None
        self.valid_tokenized_dataset = None 

        self.train_args = None
        self.trainer = None


    def _load_model(self, model_path):
        if model_path:
            self.model_path = model_path
        else:
            self.model_path = self.config["MODEL_PATH"]

        self.tokenizer = AutoTokenizer.from_pretrained(self.config["MODEL_PATH"])
        
        if self.config["USE_SPECIAL_TOKENS"]:
            special_tokens_dict = {"additional_special_tokens": ["[USER]","[ADVISOR]"]}
            num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.model = AutoModelForTokenClassification.from_pretrained(self.config["MODEL_PATH"])


    def _check_have_pre(self):
        per_cnt = 0
        for label in self.label_list:
            if "PER" in label:
                per_cnt += 1
        if per_cnt == 1:
            return False
        elif per_cnt > 1:
            return True


    def _get_labels(self, data):

        if self.config["HAVE_PRE"]:
            pre_tag = self.config["PRE_LABEL"]
            cen_tag = self.config["CEN_LABEL"]
        else:
            pre_tag = self.config["DEFAULT_LABEL"]
            cen_tag = self.config["DEFAULT_LABEL"]
        
        for item in data:
            order = item["order"]
            label_list = item["label"]
            item["flat_order"] = []
            item["flat_label"] = []
            for sentence, labels in zip(order, label_list):
                if sentence[0]:
                    text = "[USER] "
                else:
                    text = "[ADVISOR] "
                text += sentence[1].strip()
                text = text.split()
                item["flat_order"].append(text)
                label_flattened = [ "O" for i in range(len(text))]
                for label in labels:
                    for label_index in range(len(label)):
                        if label_index==0:
                            tag = pre_tag
                        else:
                            tag = cen_tag
                        tag_index = label[label_index] + 1
                        label_flattened[tag_index] = tag
                item["flat_label"].append(label_flattened)

        return data
                


    def _init_data(self):
        valid_data_path = self.config["TRAIN_DATA_DIR"]+"valid0000.json"
        train_data_path = self.config["TRAIN_DATA_DIR"]+"train0000.json"
        with open(valid_data_path, "r") as vf:
            self.valid_data = json.loads(vf.read())
            vf.close()
        with open(valid_data_path, "r") as tf:
            self.train_data = json.loads(tf.read())
            tf.close()


    def _merge_lines(self, lines):
        result = []
        for line in lines:
            result.extend(line)
        return result

    def _init_input(self, data, window_size):
        # 这里的len_bottom就是说用来记录最终计算loss的参数的长度
        input_dict = {"tokens": [] , "labels": [], "len_bottom": []}
        for order in data:
            for line_index in range(len(order["order"])):
                is_user = order["order"][line_index][0]
                if is_user:
                    temp_tokens = [order["flat_order"][line_index]]
                    temp_labels = [order["flat_label"][line_index]]
                    tokenized_inputs_bottom = self.tokenizer(order["flat_order"][line_index], truncation=True, is_split_into_words=True)
                    input_dict["len_bottom"].append(len(tokenized_inputs_bottom["input_ids"]))
                    for i in range(window_size):
                        now_index = line_index - i - 1
                        if now_index>=0:
                            temp_tokens.append(order["flat_order"][now_index])
                            temp_labels.append(order["flat_label"][now_index])
                    temp_tokens.reverse()
                    temp_labels.reverse()
                    max_sequence = self.config["MAX_SEQUENCE"]
                    if len(temp_tokens)>=max_sequence:
                        temp_tokens = temp_tokens[max_sequence:]
                        temp_labels = temp_labels[max_sequence:]
                    input_dict["tokens"].append(self._merge_lines(temp_tokens))
                    input_dict["labels"].append(self._merge_lines(temp_labels))
        return Dataset.from_pandas(pd.DataFrame({'tokens': input_dict["tokens"], 
                                                 'ner_tags': input_dict["labels"],
                                                 'len_bottom': input_dict["len_bottom"]}))


    def init_dataset(self, window_size):
        self.valid_dataset = self._init_input(self.valid_data, window_size)
        self.train_dataset = self._init_input(self.train_data, window_size)


    def _tokenize_and_align_labels(self, examples):
        label_all_tokens = True
        tokenized_inputs = self.tokenizer(list(examples["tokens"]), truncation=True, is_split_into_words=True)
        labels = []
        len_bottom = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif label[word_idx] == 'O':
                    label_ids.append(self.label_encoding_dict["O"])
                elif word_idx != previous_word_idx:
                    label_ids.append(self.label_encoding_dict[label[word_idx]])
                else:
                    label_ids.append(self.label_encoding_dict[label[word_idx]] if label_all_tokens else -100)
                previous_word_idx = word_idx
            labels.append(label_ids)
            # 统计bottom的长度
            len_bottom.append(examples["len_bottom"][i])
        tokenized_inputs["labels"] = labels
        tokenized_inputs["len_bottom"] = len_bottom
        return tokenized_inputs


    def tokenize_and_align_labels(self):
        self.train_tokenized_datasets = self.train_dataset.map(self._tokenize_and_align_labels, batched=True)
        self.valid_tokenized_datasets = self.valid_dataset.map(self._tokenize_and_align_labels, batched=True)
        json_train = json.dumps(self.train_tokenized_datasets, indent=2)
        json_valid = json.dumps(self.valid_tokenized_datasets, indent=2)
        with open("train_data/datasets/train.json") as tf:
            tf.write(json_train)
            tf.close()
        with open("train_data/datasets/valid.json") as vf:
            vf.write(json_valid)
            vf.close()
        import pdb;pdb.set_trace()

    def _compute_metrics(self, p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]


        results = self.metric.compute(predictions=true_predictions, references=true_labels)
        return_entity_level_metrics = True
        if return_entity_level_metrics:
            # Unpack nested dictionaries
            final_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f"{key}_{n}"] = v
                else:
                    final_results[key] = value
            return final_results
        else:
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }
    def get_trainer(self):
        self.train_args = TrainingArguments(
            output_dir="test-ner",
            evaluation_strategy = "steps",
            logging_dir="./test-ner/runs",
            logging_strategy="steps",
            learning_rate=self.config["lr"],
            per_device_train_batch_size=self.config["batch_size"],
            per_device_eval_batch_size=self.config["batch_size"],
            num_train_epochs=10,
            weight_decay=self.config["weight_decay"],
            fp16=True,
            half_precision_backend=True,
            # keep_batchnorm_fp32=False,
            # fp16_backend=True,
            fp16_full_eval=True,
            fp16_opt_level="O2",
            save_steps=500,
            eval_steps=10,
            logging_steps=10,
            logging_first_step=True,
            )
        self.data_collator = DataCollatorForTokenClassification(self.tokenizer)

        self.trainer = CustomerTrainer(
            self.model,
            self.train_args,
            train_dataset=self.train_tokenized_datasets,
            eval_dataset=self.valid_tokenized_datasets,
            # train_dataset=self.train_tokenized_datasets if self.train_args.do_train else None,
            # eval_dataset=self.valid_tokenized_datasets if self.train_args.do_eval else None,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self._compute_metrics,
            )

        # self.trainer = Trainer(
        #     self.model,
        #     self.train_args,
        #     train_dataset=self.train_tokenized_datasets,
        #     eval_dataset=self.valid_tokenized_datasets,
        #     # train_dataset=self.train_tokenized_datasets if self.train_args.do_train else None,
        #     # eval_dataset=self.valid_tokenized_datasets if self.train_args.do_eval else None,
        #     data_collator=self.data_collator,
        #     tokenizer=self.tokenizer
        #     )
    

def main():
    model_list = [ 
                  "xlm-roberta-large-finetuned-conll03-english",
                  "dslim/bert-base-NER",  
                  "dslim/bert-large-NER",
                  "vlan/bert-base-multilingual-cased-ner-hrl",
                  "dbmdz/bert-large-cased-finetuned-conll03-english",
                  "Jean-Baptiste/roberta-large-ner-english",
                  "51la5/bert-large-NER", 
                  "gunghio/distilbert-base-multilingual-cased-finetuned-conll2003-ner"
                 ]
    window_size = 1
    with open("config_train.yaml", "r") as configf:
        config = yaml.safe_load(configf)
    config["MODEL_PATH"] = "dslim/bert-large-NER"
    mytrainer = MyTrainer(config)
    mytrainer.init_dataset(window_size)
    mytrainer.tokenize_and_align_labels()
    mytrainer.get_trainer()
    mytrainer.trainer.train()
    mytrainer.trainer.evaluate()
    mytrainer.trainer.save_model("test.model")


if __name__=="__main__":
    main()


