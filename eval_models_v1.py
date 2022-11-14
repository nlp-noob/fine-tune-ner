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
        self.inputs = None
        self.badcase = []

        
    def _get_data_from_json(self):
        data_dict = {}
        file_name = self.config["DATA_FILE_PATH"]
        with open(file_name, 'r') as jf:
            data = json.loads(jf.read())
        return data


    def _get_label_O(self, lenth):
        label_list = []
        for i in range(lenth):
            label_list.append("O")
        return label_list
            

    def _align_labels(self):
        if self.config["HAVE_PRE"]:
            pre_label = self.config["PRE_LABEL"]
            cen_label = self.config["CEN_LABEL"]
        else:
            pre_label = self.config["DEFAULT_LABEL"]
            sen_label = self.config["DEFAULT_LABEL"]
        for item in self.data:
            aligned_label_list = []
            for label, sentence in zip(item["label"], item["order"]):
                head_inputs = self.tokenizer(sentence[0], 
                                             add_special_tokens=False, 
                                             return_tensors="pt")
                tail_inputs = self.tokenizer(sentence[1],
                                             add_special_tokens=False,
                                             return_tensors="pt")
                head_word_ids = head_inputs.word_ids()
                tail_word_ids = tail_inputs.word_ids()
                head_tokenized_sentence = self.tokenizer.convert_ids_to_tokens(head_inputs["input_ids"][0])
                tail_tokenized_sentence = self.tokenizer.convert_ids_to_tokens(tail_inputs["input_ids"][0])
                # 尾部添加label
                head_labels = self._get_label_O(len(head_word_ids))
                tail_labels = self._get_label_O(len(tail_word_ids))
                
                for a_label in label:
                    for label_index in range(len(a_label)):
                        if label_index==0:
                            label_to_tag = pre_label
                        else:
                            label_to_tag = cen_label
                        for word_id, tag_index in zip(tail_word_ids, range(len(tail_labels))):
                            if word_id==a_label[label_index]:
                                tail_labels[tag_index] = label_to_tag
                                
                head_labels.extend(tail_labels)
                aligned_label_list.append(head_labels)
            item["label"] = aligned_label_list


    def collate_inputs_Win(self):
        # 创建输入的函数
        input_list = []
        for item in self.data:
            for sentence_index in range(len(item["order"])):

                if item["order"][sentence_index][0]=="[USER]":
                    input_index = []
                    for i in range(self.config["SLIDING_WIN_SIZE"]):
                        up_index = i + 1
                        if(sentence_index-up_index<0):
                            break
                        else:
                            input_index.append(sentence_index-up_index)
                    input_index.reverse()

                    a_input = []
                    a_label = []
                    for index in input_index:
                        a_input.extend(item["order"][index])
                        a_label.extend(item["label"][index])
                    a_input = " ".join(a_input)

                    input_list.append({"input": a_input,
                                       "tag": a_label,
                                       "orderNO": item["orderNO"],
                                       "pairNO": item["pairNO"],
                                       "overlap": True})
        self.inputs = input_list


    def collate_inputs_Win_no_overlaps(self):
        # 创建输入的函数
        input_list = []
        for item in self.data:
            for sentence_index in range(len(item["order"])):

                if item["order"][sentence_index][0]=="[USER]":
                    input_index = []
                    for i in range(self.config["SLIDING_WIN_SIZE"]):
                        up_index = i + 1
                        if(sentence_index-up_index<0):
                            break
                        else:
                            input_index.append(sentence_index-up_index)
                    input_index.reverse()
                    check_list = input_index[:-1]
                    jump_flag = False
                    for check_index in check_list:
                        if item["order"][check_index][0]=="[USER]":
                            jump_flag = True
                    if jump_flag:
                        continue

                    a_input = []
                    a_label = []
                    for index in input_index:
                        a_input.extend(item["order"][index])
                        a_label.extend(item["label"][index])
                    a_input = " ".join(a_input)

                    input_list.append({"input": a_input,
                                       "tag": a_label,
                                       "orderNO": item["orderNO"],
                                       "pairNO": item["pairNO"],
                                       "overlap": False})
        self.inputs = input_list


    def get_predict_label(self):
        for a_input in self.inputs:
            tokenized_sentence = self.tokenizer(a_input["input"],
                                                add_special_tokens = self.config["ADD_SPECIAL_TOKENS"],
                                                return_tensors="pt")
            with torch.no_grad():
                logits = self.model(**tokenized_sentence).logits
            
            predicted_token_class_ids = logits.argmax(-1)
            predicted_tokens_classes = [self.model.config.id2label[t.item()] for t in predicted_token_class_ids[0]]
            if(self.config["ADD_SPECIAL_TOKENS"]):
                del(predicted_tokens_classes[-1])
                del(predicted_tokens_classes[0])
            for index in range(len(predicted_tokens_classes)):
                a_tag = predicted_tokens_classes[index]
                if(a_tag!=self.config["PRE_LABEL"] and
                   a_tag!=self.config["CEN_LABEL"] and
                   a_tag!=self.config["DEFAULT_LABEL"]):
                    predicted_tokens_classes[index] = "O"
            a_input["pred"] = predicted_tokens_classes
            a_input["tokenized_words"] = self.tokenizer.convert_ids_to_tokens(tokenized_sentence["input_ids"][0])


    def eval(self):
        True_P = 0
        False_P = 0
        False_N = 0
        for item in self.inputs:
            per_y_true = item["tag"]
            per_y_pred = item["pred"]
            list_badcase = []
            write_bad_case_flag = False
            for true_label, pred_label in zip(per_y_true, per_y_pred):
                if(true_label!='O' and true_label==pred_label):
                    list_badcase.append("TF")
                    True_P += 1
                elif(pred_label!='O' and true_label!=pred_label):
                    write_bad_case_flag = True
                    list_badcase.append("FP")
                    False_P += 1
                elif(pred_label=='O' and true_label!='O'):
                    write_bad_case_flag = True
                    list_badcase.append("FN")
                    False_N += 1
                else:
                    list_badcase.append("O")
            if write_bad_case_flag:
                    self.badcase.append(item["input"])
                    self.badcase.append(item["tokenized_words"])
                    self.badcase.append(item["tag"])
                    self.badcase.append(item["pred"])
                    self.badcase.append("**"*20)
        print("**"*20)
        print("Overlap:  "+str(self.inputs[0]["overlap"]))
        print("The Window SIZE is:  "+str(self.config["SLIDING_WIN_SIZE"]))
        print("TF = "+str(True_P))
        print("FP = "+str(False_P))
        print("FN = "+str(False_N))
        Precision = True_P/(True_P + False_P)
        Recall = True_P/(True_P + False_N)
        F1 = 2*Precision*Recall/(Precision+Recall)
        print("The Precision is:\t {}".format(Precision))
        print("The Recall is:\t\t {}".format(Recall))
        print("The F1 is:\t\t {}".format(F1))
        print("**"*20)


    def write_badcase(self):
        model_path = "_".join(self.config["MODEL_PATH"].split("/"))
        win_size = self.config["SLIDING_WIN_SIZE"]
        if self.inputs[0]["overlap"]:
            overlap = "overlap"
        else:
            overlap = "no_overlap"
        with open("badcases/{}_Win{}_{}.txt".format(model_path, win_size, overlap),  "w") as bf:
            for line in self.badcase:
                bf.write("  ".join(line)+"\n")
            
                            
def main():

    with open("config.yaml","r") as stream:
        config = yaml.safe_load(stream)
    win_size = config["SLIDING_WIN_SIZE"]
    for the_size in range(win_size):
        
        config["SLIDING_WIN_SIZE"] = the_size + 1 
        evaluator = Evaluator(config)
    
        evaluator.collate_inputs_Win()
        evaluator.get_predict_label()
        evaluator.eval()
        evaluator.write_badcase()

        evaluator.collate_inputs_Win_no_overlaps()
        evaluator.get_predict_label()
        evaluator.eval()
        evaluator.write_badcase()


if __name__=="__main__":
    main()

