import os
import re
import time
import json
import yaml
import torch
import time
from transformers import AutoTokenizer, AutoModelForTokenClassification
from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score
from seqeval.metrics import classification_report


class Evaluator:

    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["MODEL_PATH"])
        self.model = AutoModelForTokenClassification.from_pretrained(self.config["MODEL_PATH"])
        # 判定是否使用special tokens来表示角色
        if self.config["USE_SPECIAL_TOKENS"]:
            special_tokens_dict = {"additional_special_tokens": ["[USER]","[ADVISOR]"]}
            num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.data = self._get_data_from_json()
        self.total_label = 0
        self._align_labels()
        self.inputs = None
        self.badcase = []
        self.predict_time = 0

        
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
            cen_label = self.config["DEFAULT_LABEL"]
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
                tail_labels = []
                before_tokenized_sentence = sentence[1].split()
                change_tail_labels_index = 0
                for word_index in range(len(before_tokenized_sentence)):
                    tokenized_word = self.tokenizer(before_tokenized_sentence[word_index], 
                                                    add_special_tokens=False,
                                                    return_tensors="pt")
                    label_to_tag = "O"
                    get_label_flag = False

                    for a_label in label:
                        for label_index in range(len(a_label)):
                            if label_index==0:
                                label_to_tag = pre_label
                            else:
                                label_to_tag = cen_label
                            if word_index==a_label[label_index]:
                                    self.total_label += 1
                                    get_label_flag = True
                                    break
                            else:
                                label_to_tag = "O"
                        if(get_label_flag):
                            break

                    for i in range(len(tokenized_word["input_ids"][0])):
                        tail_labels.append(label_to_tag)
                                
                head_labels.extend(tail_labels)
                aligned_label_list.append(head_labels)
            item["label"] = aligned_label_list


    def collate_inputs_All(self):
        input_list = []
        for item, item_index in zip(self.data, range(len(self.data))):
            for sentence_index in range(len(item["order"])):
                input_index = [sentence_index]
                user_index = [] 
                for i in range(self.config["SLIDING_WIN_SIZE"]):
                    up_index = i + 1
                    if(sentence_index-up_index<0):
                        break
                    else:
                        input_index.append(sentence_index-up_index)
                    if(item["order"][sentence_index-up_index][0]=="[USER]"):
                        user_index.append(sentence_index-up_index)
                input_index.reverse()
                user_index.reverse()
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
                                   "overlap": True,
                                   "dataIndex": item_index, 
                                   "inputIndex": input_list,
                                   "userIndex": user_index,
                                   "inputType": "_input_ALL_"})
        self.inputs = input_list

    def collate_inputs_Only_User(self):
        input_list = []
        for item, item_index in zip(self.data, range(len(self.data))):
            for sentence_index in range(len(item["order"])):
                if item["order"][sentence_index]!="[USER]":
                    continue
                input_index = [sentence_index]
                # 这里第一个就是USER的index
                user_index = [sentence_index] 
                for i in range(self.config["SLIDING_WIN_SIZE"]):
                    up_index = i + 1
                    if(sentence_index-up_index<0):
                        break
                    else:
                        input_index.append(sentence_index-up_index)
                    if(item["order"][sentence_index-up_index][0]=="[USER]"):
                        user_index.append(sentence_index-up_index)
                input_index.reverse()
                user_index.reverse()
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
                                   "overlap": True,
                                   "dataIndex": item_index, 
                                   "inputIndex": input_list,
                                   "userIndex": user_index,
                                   "inputType": "_Only_USER_"})
        self.inputs = input_list

    def get_predict_label(self):
        start_time = time.time()
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
        end_time = time.time()
        self.predict_time = end_time - start_time


    def _format_out_put(self, word_list):
        out_str = ""
        for word in word_list:
             tab_num = int(len(word)/4)
             out_str += word+"\t"*(4-tab_num) 
        return out_str

    

    def eval_all(self):
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
                    self.badcase.append("pairNO:"+str(item["pairNO"]))
                    self.badcase.append("orderNO:"+str(item["orderNO"]))
                    self.badcase.append(self._format_out_put(item["tokenized_words"]))
                    self.badcase.append("true_label: \t"+self._format_out_put(item["tag"]))
                    self.badcase.append("pred_lable: \t"+self._format_out_put(item["pred"]))
                    self.badcase.append("**"*20)
            
        print("model_path:"+self.config["MODEL_PATH"])
        print("**"*20)
        print("Overlap:  "+str(self.inputs[0]["overlap"]))
        print("The Window SIZE is:  "+str(self.config["SLIDING_WIN_SIZE"]))
        print("There are totally {} tags.".format(self.total_label))
        print("TF = "+str(True_P)+"\t\t"+"FP = "+str(False_P)+"\t\t"+"FN = "+str(False_N))
        Precision = True_P/(True_P + False_P)
        Recall = True_P/(True_P + False_N)
        F1 = 2*Precision*Recall/(Precision+Recall)
        print("The Precision is:{}".format(Precision), end="\t")
        print("The Recall is:{}".format(Recall), end="\t")
        print("The F1 is:{}".format(F1))
        print("Using time {} in prediction".format(self.predict_time))
        print("**"*20)

        model_path = "_".join(self.config["MODEL_PATH"].split("/"))
        if self.config["USE_SPECIAL_TOKENS"]:
            model_path = model_path +"_S"
        if self.config["SLIDING_WIN_SIZE"]==0:
            logf = open("log/{}.txt".format(model_path), "w")
        else: 
            logf = open("log/{}.txt".format(model_path), "a")
        logf.write("**"*20+"\n")
        logf.write("Overlap:  "+str(self.inputs[0]["overlap"])+"\n")
        logf.write("The Window SIZE is:  "+str(self.config["SLIDING_WIN_SIZE"])+"\n")
        logf.write("There are totally {} tags.".format(self.total_label)+"\n")
        logf.write("TF = "+str(True_P)+"\t\t"+"FP = "+str(False_P)+"\t\t"+"FN = "+str(False_N)+"\n")
        logf.write("Using time {} in prediction".format(self.predict_time)+"\n")
        logf.write("The Precision is:{}".format(Precision)+"\n")
        logf.write("The Recall is:{}".format(Recall)+"\n")
        logf.write("The F1 is:{}".format(F1)+"\n")
        logf.write("**"*20+"\n")
        logf.close()


    def write_badcase(self):
        model_path = "_".join(self.config["MODEL_PATH"].split("/"))
        win_size = self.config["SLIDING_WIN_SIZE"]
        if self.inputs[0]["overlap"]:
            overlap = "overlap"
        else:
            overlap = "no_overlap"
        if self.config["USE_SPECIAL_TOKENS"]:
            overlap = overlap + "_S"
        with open("badcases/{}_Win{}_{}.txt".format(model_path, win_size, overlap),  "w") as bf:
            for line in self.badcase:
                bf.write(line+"\n")
            bf.close()

# 检测模型是否可用，若可用，通过这个函数提取模型中符合这个人名实体提取任务的标签，设置到config相应的参数中
def modify_config(model_path, config_yaml):
    try:
        model = AutoModelForTokenClassification.from_pretrained(model_path)
    except:
        print("Something went Wrong with the network!!")
        return False, config_yaml
    else:
        config_yaml["MODEL_PATH"] = model_path
        all_label = []
        for id_label in range(len(model.config.id2label)):
            label_to_append = model.config.id2label[id_label]
            if "PER" in label_to_append:
                all_label.append(label_to_append)
        if config_yaml["PRE_LABEL"] in all_label and config_yaml["CEN_LABEL"] in all_label:
            config_yaml["HAVE_PRE"] = True
            return True, config_yaml
        elif(len(all_label)==0):
            return False, config_yaml
        else:
            config_yaml["HAVE_PRE"] = False
            config_yaml["DEFAULT_LABEL"] = all_label[0]
            return True, config_yaml

                            
def main():

    tested_list = [
                  ] 
    no_net_work = [
                  "jplu/tf-xlm-r-ner-40-lang",
                  ]
    model_list = [ 
                  "dslim/bert-base-NER",  
                  "dslim/bert-large-NER",
                  "vlan/bert-base-multilingual-cased-ner-hrl",
                  "dbmdz/bert-large-cased-finetuned-conll03-english",
                  "xlm-roberta-large-finetuned-conll03-english",
                  "Jean-Baptiste/roberta-large-ner-english",
                  "cmarkea/distilcamembert-base-ner",
                  "51la5/bert-large-NER", 
                  "gunghio/distilbert-base-multilingual-cased-finetuned-conll2003-ner"
                 ]

    with open("config.yaml","r") as stream:
        config = yaml.safe_load(stream)
    for path in model_list:
        jump_flag, config = modify_config(path, config)
        if not jump_flag:
            print("Jump the path:{}. no related label".format(path))
            continue
        win_size = config["SLIDING_WIN_SIZE"]
        for the_size in range(win_size+1):
            config["SLIDING_WIN_SIZE"] = the_size
            evaluator = Evaluator(config)
        
            evaluator.collate_inputs_All()
            evaluator.get_predict_label()
            evaluator.eval_all()
            evaluator.write_badcase()

            evaluator.collate_inputs_Only_User()
            evaluator.get_predict_label()
            evaluator.eval_all()
            evaluator.write_badcase()


if __name__=="__main__":
    main()

