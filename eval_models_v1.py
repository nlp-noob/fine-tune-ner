import os
import re
import time
import json
import yaml
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score
from seqeval.metrics import classification_report

class Evaluator:

    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["MODEL_PATH"])
        self.model = AutoModelForTokenClassification.from_pretrained(self.config["MODEL_PATH"])
        self.all_conversations, self.all_labels = self._get_data_from_json()
        self._align_labels()

        
    def _get_data_from_json(self):
        file_name = self.config["DATA_FILE_PATH"]
        with open(file_name, 'r') as jf:
            data = json.loads(jf.read())
            conversation_list = []
            conversation = []
            label_list = []
            for item in data:
                for sentence in item["order"]:
                    n_sentence = []
                    n_sentence.append(sentence[0])
                    n_sentence.append(":") 
                    n_sentence.extend(sentence[1].split())
                    conversation.append(n_sentence)
                conversation_list.append(conversation)
                conversation = []
                label_list.append(item["label"])
        return conversation_list, label_list

    def _align_labels(self):
        for conversation, label in zip(self.all_conversations, self.all_labels):
            
            import pdb; pdb.set_trace()
            

    def _align_label(self, conversation, label):
        pass


def main():
    with open("config.yaml","r") as stream:
        config = yaml.safe_load(stream)
    evaluator = Evaluator(config)


if __name__=="__main__":
    main()







