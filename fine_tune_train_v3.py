#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for token classification.
"""
# You can also adapt this script on your own token classification task and datasets. Pointers for this are left as
# comments.

import logging
import os
import sys
import json
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import datasets
import numpy as np
from datasets import ClassLabel, load_dataset
from datasets import Dataset

import torch
import evaluate
import transformers
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    HfArgumentParser,
    PretrainedConfig,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import (
    EvalLoopOutput,
    EvalPrediction,
    get_last_checkpoint,
    has_length,
    denumpify_detensorize,
    find_executable_batch_size,
    )
from transformers.trainer_pt_utils import (
    find_batch_size, 
    nested_concat, 
    nested_numpify, 
    nested_truncate,
    )
from transformers.utils import (
    check_min_version, 
    send_example_telemetry, 
    is_torch_tpu_available,
    is_sagemaker_mp_enabled,
    )
from transformers.optimization import get_scheduler
from transformers.utils.versions import require_version
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Union, Any
# for drawing pr curve
from sklearn.metrics import precision_recall_curve, roc_curve
import sklearn
import matplotlib.pyplot as plt

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from .trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

#### 重写trainer方法里面的log控制输出结果
class CustomTrainer(Trainer):

    write_badcases_or_not = False
    badcases_dir = None
    input_window_size = None
    eval_file_name = None
    # 存储得到的PR曲线相应的值
    curve_log_history = []
    draw_curve_or_not = None
    curve_save_dir = None
    save_curve_step = None
    token_level_PR_content = None
    token_level_PR_auc = None
    token_level_AUC = None
    # To save best model for some specific evaluation value
    best_model_dir = None
    save_my_best_model_or_not = None
    best_metrics_keys_list = None
    # a dict that save the best metrics while training, the dict's keys is given by the list above
    best_metrics = None
    # the learning rate schedule
    lr_schedule_name = None
    # specify that use special tokens to represent the "[USER]" and "[ADVISOR]" or not
    use_special_tokens_or_not = None

    def specify_custom_args(
                            self, 
                            badcases_dir, 
                            window_size, 
                            eval_file_name,
                            draw_curve_or_not,
                            curve_save_dir,
                            save_curve_step,
                            best_model_dir,
                            save_my_best_model_or_not,
                            best_metrics_keys_list,
                            lr_schedule_name,
                            use_special_tokens_or_not,
                            ):
        self.badcases_dir = badcases_dir
        self.write_badcases_or_not = True
        self.input_window_size = window_size
        self.eval_file_name = eval_file_name
        self.draw_curve_or_not = draw_curve_or_not
        self.curve_save_dir = curve_save_dir
        self.save_curve_step = save_curve_step
        self.best_model_dir = best_model_dir
        self.save_my_best_model_or_not = save_my_best_model_or_not
        self.best_metrics_keys_list = best_metrics_keys_list.split(",")
        self.lr_schedule_name = lr_schedule_name
        self.use_special_tokens_or_not = use_special_tokens_or_not

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)
        output = {**logs, **{"step": self.state.global_step}}
        # 在这里对log进行过滤


        #if self.state.global_step%10==0:
        #    import pdb;pdb.set_trace()
        label_list = list(self.model.config.label2id.keys())
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)
    
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method (or `create_optimizer` and/or
        `create_scheduler`) in a subclass.
        """
        self.create_optimizer()
        if IS_SAGEMAKER_MP_POST_1_10 and smp.state.cfg.fp16:
            # If smp >= 1.10 and fp16 is enabled, we unwrap the optimizer
            optimizer = self.optimizer.optimizer
        else:
            optimizer = self.optimizer
        self.create_scheduler(num_training_steps=num_training_steps, optimizer=optimizer)

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )
        # this is my method to deal with the problem 
        # that the scheduler is not updated when the training step is double because of the memory size
        else:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )
        return self.lr_scheduler

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:

            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None, inference=True
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [args.device]).per_device_loader(args.device)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        inputs_host = None

        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        input_ids_for_badcases = []
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            input_ids_for_badcases.append(inputs["input_ids"])
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            inputs_decode = self._prepare_input(inputs["input_ids"]) if args.include_inputs_for_metrics else None

            if is_torch_tpu_available():
                xm.mark_step()

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            if inputs_decode is not None:
                inputs_decode = self._pad_across_processes(inputs_decode)
                inputs_decode = self._nested_gather(inputs_decode)
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                )
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if inputs_host is not None:
                    inputs_decode = nested_numpify(inputs_host)
                    all_inputs = (
                        inputs_decode
                        if all_inputs is None
                        else nested_concat(all_inputs, inputs_decode, padding_index=-100)
                    )
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, inputs_host, labels_host = None, None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
            all_inputs = (
                inputs_decode if all_inputs is None else nested_concat(all_inputs, inputs_decode, padding_index=-100)
            )
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)
        if all_inputs is not None:
            all_inputs = nested_truncate(all_inputs, num_samples)

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            if args.include_inputs_for_metrics:
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs)
                )
            else:
                metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}

        # get the pr and orc curve
        if self.draw_curve_or_not:
            self.draw_the_curve(all_preds, all_labels)
        # 添加曲线的参数
        metrics["PR_content"] = self.token_level_PR_content
        metrics["PR_auc"] = self.token_level_PR_auc
        metrics["ROC_auc"] = self.token_level_AUC

        # 比较参数并保存最好的模型和指标
        if self.save_my_best_model_or_not:
            self.save_my_best_model_and_eval(metrics)

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)


        # Write My Bad case in eval process
        if self.write_badcases_or_not:
            tokenized_word_list = [] 
            for a_input in input_ids_for_badcases:
                temp_list = []
                for a_word_id in a_input:
                    for a_token_id in a_word_id:
                        temp_list.append(self.tokenizer.decode(a_token_id))
                    tokenized_word_list.append(" ".join(temp_list))
                    temp_list = []
            self.write_badcases(all_preds, all_labels, tokenized_word_list)
            tokenized_word_list = None
            input_ids_for_badcases = None
        ##################

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def _find_need_keys_in_metrics(self, metrics):
        metrics_key_list = list(metrics.keys())
        if len(self.best_metrics_keys_list)==0:
            return metrics
        result_metrics = {}
        for key in self.best_metrics_keys_list:
            if key in metrics_key_list:
                result_metrics[key] = metrics[key]
            else:
                continue
        return result_metrics

    def save_my_best_model_and_eval(self, metrics):
        need_metrics = self._find_need_keys_in_metrics(metrics)
        key_list = list(need_metrics.keys())
        if self.best_metrics is None:
            self.best_metrics = {}
            self.best_metrics["step"] = self.state.global_step
            for key in self.best_metrics_keys_list:
                self.best_metrics["best_"+key] = need_metrics
                for key2 in self.best_metrics_keys_list:
                    if key2 not in key_list:
                        self.best_metrics["best_"+key][key2] = 0
        else:
            self.best_metrics["step"] = self.state.global_step
            for key in key_list:
                train_model_name = "_".join(self.model.config.name_or_path.split("/"))
                if self.use_special_tokens_or_not:
                    train_model_name += "_s"
                dir_name = "{}/{}_win{}_{}".format(self.best_model_dir, train_model_name, self.input_window_size, self.lr_schedule_name)
                model_name = key + "_best.model"
                if not os.path.exists(dir_name):
                    os.mkdir(dir_name)
                if self.best_metrics["best_"+key][key] < need_metrics[key]:
                    self.best_metrics["best_"+key] = need_metrics
                    self.best_metrics["steps"] = self.state.global_step
                    self.save_model(dir_name + "/" + model_name)
                    json_str = json.dumps(self.best_metrics, indent=2)
                    fout = open(dir_name + "/" + "best_metrics.json", "w")
                    fout.write(json_str)
                    fout.close()

    def _sort_for_pr_auc(self, prediction, recall):
        array_for_sort = np.array([])
        for index, pred in enumerate(prediction):
            if index == 0:
                array_for_sort = np.array([[prediction[index], recall[index]]])
            else:    
                array_for_sort = np.append(array_for_sort, [[prediction[index], recall[index]]], axis=0)
        index1 = (array_for_sort[:,1]).argsort()
        sorted_array = array_for_sort[index1]
        sorted_prediction = []
        sorted_recall = []
        for a_unit in sorted_array:
            sorted_prediction.append(a_unit[0])
            sorted_recall.append(a_unit[1])
        return sklearn.metrics.auc(sorted_recall, sorted_prediction)
        

    def draw_the_curve(self, predictions, label_ids):
        o_index = None
        for index, key in enumerate(self.model.config.label2id.keys()):
            if key=="O":
                o_index = index
        # 把softmax得到的结果转化成为最大的非O的几个标签中所对应的最大概率值集合
        # token_层面的统计
        probs = []
        labels = []
        for a_input_probs, a_input_labels in zip(predictions, label_ids):
            for a_token_probs, a_label in zip(a_input_probs, a_input_labels):
                if a_label == -100:
                    continue
                elif a_label == self.model.config.label2id["O"]:
                    label_to_append = 0
                else:
                    label_to_append = 1
                max_prob = -100
                for index, a_token_prob in enumerate(a_token_probs):
                    if index==o_index:
                        continue
                    elif(a_token_prob>=max_prob):
                        max_prob = a_token_prob
                probs.append(max_prob)
                labels.append(label_to_append)

        model_name = "_".join(self.model.config.name_or_path.split("/"))
        if self.use_special_tokens_or_not:
            model_name += "_s"
        dir_name = "{}/{}_win{}_{}".format(self.curve_save_dir, model_name, self.input_window_size, self.lr_schedule_name)
        train_step = self.state.global_step
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        # token_PR 
        precision, recall, thresholds = precision_recall_curve(labels,probs,pos_label=1)
        plt.cla()
        plt.plot(recall, precision)
        plt.title("P-R curve")
        plt.xlabel('recall')
        plt.ylabel('precision')
        if train_step % self.save_curve_step == 0:
            file_name = "PR_curve"+f"{train_step:08d}"+".png"
            f = plt.gcf()
            f.savefig(dir_name + "/" + file_name)
            f.clear()
        pr_content = 0
        for index in range(min([len(precision), len(recall)])):
            if index==0:
                continue
            else:
                pr_content+=(precision[index] + precision[index-1])*abs(recall[index] -recall[index-1])/2
        self.token_level_PR_content = pr_content
        self.token_level_PR_auc = self._sort_for_pr_auc(precision, recall)
        
        # token_ROC
        fpr,tpr,thresholds = roc_curve(labels, probs, pos_label=1)
        plt.cla()
        plt.plot(fpr,tpr)
        plt.title("ROC curve")
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        if train_step % self.save_curve_step == 0:
            file_name = "ROC_curve"+f"{train_step:08d}"+".png"
            f = plt.gcf()
            f.savefig(dir_name + "/" + file_name)
            f.clear()
        self.token_level_AUC = sklearn.metrics.auc(fpr,tpr)

    def _format_out_put(self, word_list):
        out_str = ""
        for word in word_list:
             tab_num = int(len(word)/4)
             out_str += word+"\t"*(4-tab_num) 
        return out_str

    def change_labelid_to_label(self, label_lists):
        result_label = []
        for label_list in label_lists:
            temp_list = []
            for a_label in label_list:
                if a_label==-100:
                    temp_list.append("[PAD]")
                else:
                    temp_list.append(self.model.config.id2label[a_label])
            result_label.append(temp_list)
        return result_label
                
    def _tokens_list_to_words(self, tokens_list):
        words_list = []
        for token in tokens_list:
            if token[:2]=="##":
                if len(words_list)==0:
                    words_list.append(token[2:])
                else:
                    words_list[-1] = words_list[-1] + token[2:]
            else:
                words_list.append(token)
        return words_list

    def write_badcases(self, all_preds, all_labels, words):
        pred_labels = self.change_labelid_to_label(np.argmax(all_preds, axis=2))
        true_labels = self.change_labelid_to_label(all_labels)
        badcases = [] 
        model_name = "_".join(self.model.config.name_or_path.split("/"))
        if self.use_special_tokens_or_not:
            model_name += "_s"
        dir_name = "{}/{}_win{}_{}".format(self.badcases_dir, model_name, self.input_window_size, self.lr_schedule_name)
        for pred_tags, true_tags, tokenized_word_list in zip(pred_labels, true_labels, words):
            write_badcases_flag = False
            pred_tags_to_write = pred_tags[:len(tokenized_word_list.split())]
            true_tags_to_write = true_tags[:len(tokenized_word_list.split())]
            tokens_to_write = tokenized_word_list.split()
            tokens_list = []
            for a_pred, a_true, a_token in zip(pred_tags, true_tags, tokenized_word_list.split()):
                if a_true=="[PAD]":
                    continue
                if (a_true=="O" and a_pred!="O") or (a_true!="O" and a_pred=="O"):
                    write_badcases_flag = True
                if write_badcases_flag and a_token!="[UNK]":
                    tokens_list.append(a_token)
            wrong_words = self._tokens_list_to_words(tokens_list)
            if write_badcases_flag:
                badcases.append("**"*20)
                badcases.append("all_words:\t"+"\t".join(wrong_words))
                badcases.append("origin_toks: \t"+self._format_out_put(tokenized_word_list.split()))
                badcases.append("true_label: \t"+self._format_out_put(true_tags_to_write))
                badcases.append("pred_lable: \t"+self._format_out_put(pred_tags_to_write))
                badcases.append("**"*20)
        bad_step = self.state.global_step

        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        
        file_name = "bad_case_for_step_"+f"{bad_step:08d}"+".txt"
        
        with open(dir_name + "/" + file_name, "w") as fout:
            for line in badcases:
                fout.write(line+"\n")
            fout.close()

####

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# 不知道这里为什么是这个版本，在pipversion里面最高只有24
# check_min_version("4.25.0.dev0")
check_min_version("4.24.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/token-classification/requirements.txt")

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: Optional[str] = field(default="ner", metadata={"help": "The name of the task (ner, pos...)."})
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a csv or JSON file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."},
    )
    text_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of text to input in the file (a csv or JSON file)."}
    )
    label_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of label to input in the file (a csv or JSON file)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. If set, sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    label_all_tokens: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to put the label for one word on all tokens of generated by that word or just on the "
                "one (in which case the other tokens will have a padding index)."
            )
        },
    )
    return_entity_level_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to return all the entity levels during evaluation or just the overall ones."},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise oalueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        
        self.task_name = self.task_name.lower()


@dataclass
class CustomArguments:
    use_padding_for_context: bool = field(
        default=None,
        metadata={
            "help": (
                "You should set this value to decide whether to use the padding for contex"
            )
        },
    )
    write_badcases: bool = field(
        default=None,
        metadata={
            "help": (
                "this option specify whether to write badcases or not "
            )
        },
    )
    badcases_dir: str = field(
        default=None,
        metadata={
            "help": (
                "specify the path to badcases"
            )
        },
    )
    input_window_size: int = field(
        default=None,
        metadata={
            "help": (
                "the windowsize of the input that create by the input creater."
            )
        },
    )
    draw_curve_or_not: bool = field(
        default=False,
        metadata={
            "help": (
                "If you wanna draw a PR curve in the eval process."
            )
        },
    )
    curve_save_dir: str = field(
        default=None,
        metadata={
            "help": (
                "the path that the pr curve save to."
            )
        },
    )
    save_curve_step: int = field(
        default=None,
        metadata={
            "help": (
                "the step to save curve picture."
            )
        },
    )
    best_model_dir: str = field(
        default=None,
        metadata={
            "help": (
                "specify the path to save my model"
            )
        },
    )
    save_my_best_model_or_not: bool = field(
        default=None,
        metadata={
            "help": (
                "whether to save the best model or not. may be have some effect on the training time using"
            )
        },
    )
    best_metrics_keys_list: str = field(
        default=None,
        metadata={
            "help": (
                "the key to compare in order to get the best model, please split different words in , trunc"
            )
        },
    )
    use_special_tokens_or_not: bool = field(
        default=False,
        metadata={
            "help": (
                "decide using the special tokens or not"
            )
        },
    )
    special_tokens_list: str = field(
        default=None,
        metadata={
            "help": (
                "specify the tokens that you wanna add to the embeddings please split different words in , trunc"
            )
        },
    )
    def __post_init__(self):
        if self.use_padding_for_context is None:
            raise ValueError("You should specify whether to use tje padding for label to context or not.")

#####
def _merge_lines(lines):
    result = []
    for line in lines:
        result.extend(line)
    return result

def _get_labels(data):
    pre_tag = "B-PER"
    cen_tag = "I-PER"
    for item in data:
        order = item["order"]
        my_label_list = item["label"]
        item["flat_order"] = []
        item["flat_label"] = []
        for sentence, labels in zip(order, my_label_list):
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

def get_my_dataset(path, window_size):
    # init data:
    with open(path, "r") as fin:
        data = json.loads(fin.read())
        fin.close()
    ####
    data = _get_labels(data)
    input_dict = {"tokens": [] , "labels": [], "bottom_len":[]}
    for order in data:
        for line_index in range(len(order["order"])):
            is_user = order["order"][line_index][0]
            if is_user:
                temp_tokens = [order["flat_order"][line_index]]
                temp_labels = [order["flat_label"][line_index]]
                input_dict["bottom_len"].append(len(order["flat_order"][line_index]))
                for i in range(window_size):
                    now_index = line_index - i - 1
                    if now_index>=0:
                        temp_tokens.append(order["flat_order"][now_index])
                        temp_labels.append(order["flat_label"][now_index])
                temp_tokens.reverse()
                temp_labels.reverse()
                input_dict["tokens"].append(_merge_lines(temp_tokens))
                input_dict["labels"].append(_merge_lines(temp_labels))
    return Dataset.from_pandas(pd.DataFrame({'tokens': input_dict["tokens"], 
                                             'ner_tags': input_dict["labels"],
                                             'bottom_len': input_dict["bottom_len"]}))

#####

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, CustomArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, custom_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, custom_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_ner", model_args, data_args)

    # Setup logging
    ####### my_setup
    my_model_name = model_args.config_name if model_args.config_name else model_args.model_name_or_path
    my_model_name = "_".join(my_model_name.split('/'))
    #######
    logging.basicConfig(
        format="{}%(asctime)s - %(levelname)s - %(name)s - %(message)s".format(my_model_name),
        datefmt="{}%m/%d/%Y %H:%M:%S".format(my_model_name),
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # logging.basicConfig(
    #     filename="./test-ner/runs/" + my_model_name+"/",
    # )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        import pdb; pdb.set_trace()
    else:
        # data_files = {}
        dataset_dict = {}
        if data_args.train_file is not None:
            # data_files["train"] = data_args.train_file
            dataset_dict["train"] = get_my_dataset(data_args.train_file, custom_args.input_window_size)
            # check data
            # for i, labels in enumerate(dataset_dict["train"]["ner_tags"]):
                # if "B-PER" in labels:
                #     print(dataset_dict["train"]["ner_tags"][i])
                #     print(dataset_dict["train"]["tokens"][i])
                #     import pdb; pdb.set_trace()
            
        if data_args.validation_file is not None:
            # data_files["validation"] = data_args.validation_file
            dataset_dict["validation"] = get_my_dataset(data_args.validation_file, custom_args.input_window_size)
        if data_args.test_file is not None:
            # data_files["test"] = data_args.test_file
            dataset_dict["test"] = get_my_dataset(data_args.test_file, custom_args.input_window_size)
        raw_datasets = datasets.DatasetDict(dataset_dict) 
        # extension = data_args.train_file.split(".")[-1]
        # raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)
        # import pdb; pdb.set_trace()
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
        features = raw_datasets["train"].features
    else:
        column_names = raw_datasets["validation"].column_names
        features = raw_datasets["validation"].features

    if data_args.text_column_name is not None:
        text_column_name = data_args.text_column_name
    elif "tokens" in column_names:
        text_column_name = "tokens"
    else:
        text_column_name = column_names[0]

    if data_args.label_column_name is not None:
        label_column_name = data_args.label_column_name
    elif f"{data_args.task_name}_tags" in column_names:
        label_column_name = f"{data_args.task_name}_tags"
    else:
        label_column_name = column_names[1]

    # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
    # unique labels.
    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    # If the labels are of type ClassLabel, they are already integers and we have the map stored somewhere.
    # Otherwise, we have to get the list of labels manually.
    labels_are_int = isinstance(features[label_column_name].feature, ClassLabel)
    if labels_are_int:
        label_list = features[label_column_name].feature.names
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        label_list = get_label_list(raw_datasets["train"][label_column_name])
        label_to_id = {l: i for i, l in enumerate(label_list)}

    num_labels = len(label_list)


    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    tokenizer_name_or_path = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path

    if config.model_type in {"bloom", "gpt2", "roberta"}:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=True,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            add_prefix_space=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=True,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    model = AutoModelForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )
    # use the special tokens "[USER]" "[ADVISOR]"
    if custom_args.use_special_tokens_or_not:
        special_tokens_list = custom_args.special_tokens_list.split(",")
        special_tokens_dict = {"additional_special_tokens": ["[USER]","[ADVISOR]"]}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models at"
            " https://huggingface.co/transformers/index.html#supported-frameworks to find the model types that meet"
            " this requirement"
        )

    # Model has labels -> use them.
    if model.config.label2id != PretrainedConfig(num_labels=len(label_list)).label2id:
        if list(sorted(model.config.label2id.keys())) == list(sorted(label_list)):
            # Reorganize `label_list` to match the ordering of the model.
            if labels_are_int:
                label_to_id = {i: int(model.config.label2id[l]) for i, l in enumerate(label_list)}
                label_list = [model.config.id2label[i] for i in range(num_labels)]
            else:
                label_list = [model.config.id2label[i] for i in range(num_labels)]
                label_to_id = {l: i for i, l in enumerate(label_list)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(model.config.label2id.keys()))}, dataset labels:"
                f" {list(sorted(label_list))}.\nIgnoring the model labels as a result.",
            )

    # Set the correspondences label/ID inside the model config
    model.config.label2id = {l: i for i, l in enumerate(label_list)}
    model.config.id2label = {i: l for i, l in enumerate(label_list)}

    # Map that sends B-Xxx label to its I-Xxx counterpart
    b_to_i_label = []
    for idx, label in enumerate(label_list):
        if label.startswith("B-") and label.replace("B-", "I-") in label_list:
            b_to_i_label.append(label_list.index(label.replace("B-", "I-")))
        else:
            b_to_i_label.append(idx)

    # Preprocessing the dataset
    # Padding strategy
    padding = "max_length" if data_args.pad_to_max_length else False

    # Tokenize all texts and align the labels with them.
    def tokenize_and_align_labels(examples):
        # check for bottom line
        if custom_args.use_padding_for_context:
            for i, tokens_check in enumerate(examples[text_column_name]):
                bottom_len = examples["bottom_len"][i]
                if examples["tokens"][i][-bottom_len]!="[USER]":
                    raise ValueError("wrong dataset")
            
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            padding=padding,
            truncation=True,
            max_length=data_args.max_seq_length,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )
        labels = []
        for i, label in enumerate(examples[label_column_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []

            if custom_args.use_padding_for_context:
                bottom_len = examples["bottom_len"][i]
                tokenized_bottom = tokenizer(
                    examples[text_column_name][i][-bottom_len:],
                    padding=padding,
                    truncation=True,
                    max_length=data_args.max_seq_length,
                    # We use this argument because the texts in our dataset are lists of words (with a label for each word).
                    is_split_into_words=True,
                )
                len_tokenized_bottom = len(tokenized_bottom["input_ids"]) - 2

            for j, word_idx in enumerate(word_ids):
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                elif custom_args.use_padding_for_context and j<=(len(word_ids)-len_tokenized_bottom):
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    if data_args.label_all_tokens:
                        label_ids.append(b_to_i_label[label_to_id[label[word_idx]]])
                    else:
                        label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                tokenize_and_align_labels,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                tokenize_and_align_labels,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                tokenize_and_align_labels,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)

    # Metrics
    metric = evaluate.load("seqeval")

    def _split_into_word_list(labels):
        flattened_labels = []
        for a_line_labels in labels:
            for a_label in a_line_labels:
                # 因为这个是模型的预测结果，不会再进行id映射，定义一个任意的label告诉metrics这是单独的片段即可
                if a_label!="O":
                    a_label="B-PER"
                flattened_labels.append([a_label])
        return flattened_labels

    def compute_metrics(p):
        #########
        predictions = p.predictions
        labels = p.label_ids
        #########
        # predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]


        true_pred_splitted = _split_into_word_list(true_predictions)
        true_labels_splitted = _split_into_word_list(true_labels)

        # 按照词片段
        results_piece = metric.compute(predictions=true_predictions, references=true_labels)
        # 按照标签
        results_token_level = metric.compute(predictions=true_pred_splitted, references=true_labels_splitted)
        if data_args.return_entity_level_metrics:
            # Unpack nested dictionaries
            final_results = {}
            for key, value in results_piece.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f"word_piece_{key}_{n}"] = v
                else:
                    final_results["word_piece_" + key] = value
            for key, value in results_token_level.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f"token_level_{key}_{n}"] = v
                else:
                    final_results["token_level_" + key] = value
            return final_results
        else:
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }

    # Initialize our CustomTrainer extends from Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    if custom_args.write_badcases:
        trainer.specify_custom_args(
                                        custom_args.badcases_dir, 
                                        custom_args.input_window_size,
                                        data_args.validation_file,
                                        custom_args.draw_curve_or_not,
                                        custom_args.curve_save_dir,
                                        custom_args.save_curve_step,
                                        custom_args.best_model_dir,
                                        custom_args.save_my_best_model_or_not,
                                        custom_args.best_metrics_keys_list, 
                                        training_args.lr_scheduler_type,
                                        custom_args.use_special_tokens_or_not,
                                      )
    
    print("**"*50)
    print("**"*50)
    print("**"*50)
    print("**"*50)
    print("**"*50)
    print("**"*50)
    print("**"*50)

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        # training begin
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        # training end
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # When training ended
    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")

        predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        # Save predictions
        output_predictions_file = os.path.join(training_args.output_dir, "predictions.txt")
        if trainer.is_world_process_zero():
            with open(output_predictions_file, "w") as writer:
                for prediction in true_predictions:
                    writer.write(" ".join(prediction) + "\n")

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "token-classification"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
