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
        self.data = self._get_data_from_json()
        self._align_labels()
        
    def _get_data_from_json(self):
        data_dict = {}
        file_name = self.config["DATA_FILE_PATH"]
        with open(file_name, 'r') as jf:
            data = json.loads(jf.read())
        return data

    def _align_labels(self):
        for item in self.data:
            item["align_label"] = []
        import pdb; pdb.set_trace()

    def _align_label(self, conversation, label):
        pass


def main():
    with open("config.yaml","r") as stream:
        config = yaml.safe_load(stream)
    evaluator = Evaluator(config)


if __name__=="__main__":
    main()







