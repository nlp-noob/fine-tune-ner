import os
import re
import time
import json
import yaml
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification


class Trainer:

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config["device"])
        self.model_path = None
        self.model = None
        self.tokenizer = None
        

    def load_model(self, model_path):
        if model_path:
            self.model_path = model_path
        else:
            self.model_path = self.config["MODEL_PATH"]
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["MODEL_PATH"])
        self.model = AutoModelForTokenClassification.from_pretrained(self.config["MODEL_PATH"]).to(self.device)
        
            
            




def main():
    with open("config_train.yaml", "r") as configf:
        config = yaml.safe_load(configf)
    trainer = Trainer(config)
    trainer.load_model(None)


if __name__=="__main__":
    main()


