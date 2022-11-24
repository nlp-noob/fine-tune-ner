import os
import re
import time
import json
import yaml
import torch
import pandas as pd
from datasets import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification
from transformers import TrainingArguments
from transformers import Trainer
from transformers import DataCollatorForTokenClassification

class MyTrainer:

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config["device"])

        self.model_path = None
        self.model = None
        self.tokenizer = None
        self.data_collator = None
        self._load_model(None)
        self.label_encoding_dict = {"I-PER": 6, "O": 7}

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
        self.data_collator = DataCollatorForTokenClassification(self.tokenizer)
        if self.config["USE_SPECIAL_TOKENS"]:
            special_tokens_dict = {"additional_special_tokens": ["[USER]","[ADVISOR]"]}
            num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.model = AutoModelForTokenClassification.from_pretrained(self.config["MODEL_PATH"]).to(self.device)


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
                text += sentence[1]
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
        valid_data_path = self.config["TRAIN_DATA_DIR"]+"valid.json"
        train_data_path = self.config["TRAIN_DATA_DIR"]+"train.json"
        with open(valid_data_path, "r") as vf:
            self.valid_data = json.loads(vf.read())
            vf.close()
        with open(valid_data_path, "r") as tf:
            self.train_data = json.loads(tf.read())
            tf.close()

    def _get_label_O(self, lenth):
        label_list = []
        for i in range(lenth):
            label_list.append("O")
        return label_list

    def _merge_lines(self, lines):
        result = []
        for line in lines:
            result.extend(line)
        return result

    def _init_input(self, data, window_size):
        input_dict = {"tokens": [] , "labels": []}
        for order in data:
            for line_index in range(len(order["order"])):
                is_user = order["order"][line_index][0]
                if is_user:
                    temp_tokens = [order["flat_order"][line_index]]
                    temp_labels = [order["flat_label"][line_index]]
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
        return Dataset.from_pandas(pd.DataFrame({'tokens': input_dict["tokens"], 'ner_tags': input_dict["labels"]}))


    def init_dataset(self, window_size):
        self.valid_dataset = self._init_input(self.valid_data, window_size)
        self.train_dataset = self._init_input(self.train_data, window_size)


    def _tokenize_and_align_labels(self, examples):
        label_all_tokens = True
        tokenized_inputs = self.tokenizer(list(examples["tokens"]), truncation=True, is_split_into_words=True)
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif label[word_idx] == '0':
                    label_ids.append(0)
                elif word_idx != previous_word_idx:
                    label_ids.append(self.label_encoding_dict[label[word_idx]])
                else:
                    label_ids.append(self.label_encoding_dict[label[word_idx]] if label_all_tokens else -100)
                previous_word_idx = word_idx
    
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs


    def tokenize_and_align_labels(self):
        self.train_tokenized_datasets = self.train_dataset.map(self._tokenize_and_align_labels, batched=True)
        self.valid_tokenized_datasets = self.valid_dataset.map(self._tokenize_and_align_labels, batched=True)

    def get_trainer(self):
        self.train_args = TrainingArguments(
            f"test-ner",
            evaluation_strategy = "epoch",
            learning_rate=self.config["lr"],
            per_device_train_batch_size=self.config["batch_size"],
            per_device_eval_batch_size=self.config["batch_size"],
            num_train_epochs=3,
            weight_decay=self.config["weight_decay"],
            )
        self.trainer = Trainer(
            self.model,
            self.train_args,
            train_dataset=self.train_tokenized_datasets,
            eval_dataset=self.valid_tokenized_datasets,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            )
    

def main():
    window_size = 1
    with open("config_train.yaml", "r") as configf:
        config = yaml.safe_load(configf)
    mytrainer = MyTrainer(config)
    mytrainer.init_dataset(window_size)
    mytrainer.tokenize_and_align_labels()
    mytrainer.get_trainer()
    mytrainer.trainer.train()
    mytrainer.trainer.evaluate()
    mytrainer.trainer.save_model(test.model)


    import pdb;pdb.set_trace()


if __name__=="__main__":
    main()


