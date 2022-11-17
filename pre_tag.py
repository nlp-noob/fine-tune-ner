import os
import re
import time
import json
import yaml
import torch
import time
from transformers import AutoTokenizer, AutoModelForTokenClassification

class PreTagger:

    def __init__(self, model_path, data_path, dir_path):
        self.device = torch.device("cuda")
        self.model_path = model_path
        self.dir_path = dir_path
        self.data_path = data_path
        self.data = self._get_data_from_json()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_path).to(self.device)

    

    def _get_data_from_json(self):
        with open(self.dir_path+self.data_path, "r") as jf:
            data = json.loads(jf.read())
        return data

    def _get_predict_label(self):

        for order, order_index in zip(self.data, range(len(self.data))):
            if order_index%50==0:
                print("processing{}/{}".format(order_index, len(self.data)))
            order["pred_label"] = []
            order["label"] = []
            for line in order["order"]:
                head_text = line[0]
                tail_text = line[1]
                text = " ".join(line)
                tokenized_sentence = self.tokenizer(text, add_special_tokens = True, return_tensors="pt").to(self.device)
                text_word_ids = self.tokenizer(text, add_special_tokens = False, return_tensors="pt").word_ids()
                head_word_ids = self.tokenizer(head_text, add_special_tokens = False, return_tensors="pt").word_ids()
                tail_word_ids = self.tokenizer(tail_text, add_special_tokens = False, return_tensors="pt").word_ids()
                with torch.no_grad():
                    logits = self.model(**tokenized_sentence).logits
                    predicted_token_class_ids = logits.argmax(-1)
                    predicted_tokens_classes = [self.model.config.id2label[t.item()] for t in predicted_token_class_ids[0]]
                    del(predicted_tokens_classes[-1])
                    del(predicted_tokens_classes[0])
                order["pred_label"].append(predicted_tokens_classes)
                tail_pred = predicted_tokens_classes[len(head_word_ids):]

                word_label = []
                # 按照词来遍历
                for word in range(tail_word_ids[-1]+1):
                    not_O_list = []
                    for word_id, pred_label in zip(tail_word_ids, tail_pred):
                        # 跳过非当前词
                        if word!=word_id:
                            continue
                        # 当前词
                        if pred_label=="O":
                            word_label.append("O")
                            break
                        else:
                            not_O_list.append(pred_label)
                    if not_O_list:
                        word_label.append(not_O_list[0])

                a_piece_indexs = []
                label_indexs = []
                for index in range(len(word_label)):
                    if word_label[index]=="O" and a_piece_indexs:
                        label_indexs.append(a_piece_indexs)
                        a_piece_indexs = []
                    elif word_label[index]!="O":
                        a_piece_indexs.append(index)
                order["label"].append(label_indexs)
            del(order["pred_label"])

    def pretag(self):
        self._get_predict_label()
        pass

    def write_data(self):
        json_data = json.dumps(self.data, indent=2)
        with open(self.dir_path+"tagged_"+self.data_path, "w") as jf:
            jf.write(json_data)
            print("write successed")
            jf.close()


def main():
    pretag_model = ["xlm-roberta-large-finetuned-conll03-english"]
    data_path = "untagged_data.json"
    dir_path = "eval_data/"
    pretagger = PreTagger(pretag_model[0], data_path, dir_path)
    pretagger.pretag()
    pretagger.write_data()


if __name__=="__main__":
    main()
