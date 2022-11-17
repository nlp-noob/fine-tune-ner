import os
import re
import time
import json
import yaml
import torch
import time
from transformers import AutoTokenizer, AutoModelForTokenClassification

class Evaluator:

    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["MODEL_PATH"])
        self.device = torch.device("cuda")
        self.model = AutoModelForTokenClassification.from_pretrained(self.config["MODEL_PATH"]).to(self.device)
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
        self._overlap_dict = {}

        
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
            item["labelword"] = item["label"].copy()
            item["label"] = aligned_label_list


    def collate_inputs_All(self):
        input_list = []
        for item, item_index in zip(self.data, range(len(self.data))):
            for sentence_index in range(len(item["order"])):
                input_index = [sentence_index]
                if(item["order"][sentence_index][0]=="[USER]"):
                    user_index = [sentence_index]
                else:
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
                word_label_list = []
                for index in input_index:
                    word_label = item["labelword"][index]
                    for a_word_label in word_label:
                        for a_word_label_index in a_word_label:
                            word_label_list.append(a_word_label_index+len(" ".join(a_input))+1)
                    a_input.extend(item["order"][index])
                    a_label.extend(item["label"][index])
                a_input = " ".join(a_input)

                input_list.append({"input": a_input,
                                   "tag": a_label,
                                   "orderNO": item["orderNO"],
                                   "pairNO": item["pairNO"],
                                   "dataIndex": item_index, 
                                   "inputIndex": input_index,
                                   "userIndex": user_index,
                                   "inputType": "_input_ALL_",
                                   "wordLabelList": word_label_list})
        self.inputs = input_list

    def collate_inputs_Only_User(self):
        input_list = []
        for item, item_index in zip(self.data, range(len(self.data))):
            for sentence_index in range(len(item["order"])):
                if item["order"][sentence_index][0]!="[USER]":
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
                word_label_list = []
                for index in input_index:
                    word_label = item["labelword"][index]
                    for a_word_label in word_label:
                        for a_word_label_index in a_word_label:
                            word_label_list.append(a_word_label_index+len(" ".join(a_input))+1)
                    a_input.extend(item["order"][index])
                    a_label.extend(item["label"][index])
                a_input = " ".join(a_input)

                input_list.append({"input": a_input,
                                   "tag": a_label,
                                   "orderNO": item["orderNO"],
                                   "pairNO": item["pairNO"],
                                   "dataIndex": item_index, 
                                   "inputIndex": input_index,
                                   "userIndex": user_index,
                                   "inputType": "_Only_USER_",
                                   "wordLabelList": word_label_list})
        self.inputs = input_list

    def get_predict_label(self):
        start_time = time.time()
        for a_input in self.inputs:
            tokenized_sentence = self.tokenizer(a_input["input"],
                                                add_special_tokens = self.config["ADD_SPECIAL_TOKENS"],
                                                return_tensors="pt").to(self.device)
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

    def _get_metrics(self, TP, FP, FN):
        Precision = TP/(TP + FP)
        Recall = TP/(TP + FN)
        F1 = 2*Precision*Recall/(Precision+Recall)
        return Precision, Recall, F1

    def write_log(self, eval_type, TP, FP, FN, TP_W, FP_W, FN_W):
        Precision, Recall, F1 = self._get_metrics(TP, FP, FN) 
        Precision_A, Recall_A, F1_A = self._get_metrics(TP_W[0], FP_W[0], FN_W[0]) 
        Precision_B, Recall_B, F1_B = self._get_metrics(TP_W[1], FP_W[1], FN_W[1]) 
        Precision_C, Recall_C, F1_C = self._get_metrics(TP_W[2], FP_W[2], FN_W[2]) 
        model_path = "_".join(self.config["MODEL_PATH"].split("/"))
        if self.config["USE_SPECIAL_TOKENS"]:
            model_path = model_path + "_" + self.inputs[0]["inputType"] + eval_type + "_S"
        else:
            model_path = model_path + "_" + self.inputs[0]["inputType"] + eval_type
        if self.config["SLIDING_WIN_SIZE"]==0:
            json_dict = {"Win": [0], 
                         "TP": [TP],
                         "FP": [FP],
                         "FN": [FN],
                         "Precision": [Precision],
                         "Recall": [Recall],
                         "F1": [F1],
                         "PredictTime": [self.predict_time],
                         "TP_A": [TP_W[0]],
                         "FP_A": [FP_W[0]],
                         "FN_A": [FN_W[0]],
                         "TP_B": [TP_W[1]],
                         "FP_B": [FP_W[1]],
                         "FN_B": [FN_W[1]],
                         "TP_C": [TP_W[2]],
                         "FP_C": [FP_W[2]],
                         "FN_C": [FN_W[2]],
                         "Precision_A": [Precision_A],
                         "Recall_A": [Recall_A],
                         "F1_A": [F1_A],
                         "Precision_B": [Precision_B],
                         "Recall_B": [Recall_B],
                         "F1_B": [F1_B],
                         "Precision_C": [Precision_C],
                         "Recall_C": [Recall_C],
                         "F1_C": [F1_C],
            
                         }
        else:
            jfr = open("log/{}.json".format(model_path), "r")
            json_dict = json.load(jfr)
            json_dict["Win"].append(self.config["SLIDING_WIN_SIZE"])
            json_dict["TP"].append(TP)
            json_dict["FP"].append(FP)
            json_dict["FN"].append(FN)
            json_dict["Precision"].append(Precision)
            json_dict["Recall"].append(Recall)
            json_dict["F1"].append(F1)
            json_dict["PredictTime"].append(self.predict_time)
            json_dict["TP_A"].append(TP_W[0])
            json_dict["FP_A"].append(FP_W[0])
            json_dict["FN_A"].append(FN_W[0])
            json_dict["TP_B"].append(TP_W[1])
            json_dict["FP_B"].append(FP_W[1])
            json_dict["FN_B"].append(FN_W[1])
            json_dict["TP_C"].append(TP_W[2])
            json_dict["FP_C"].append(FP_W[2])
            json_dict["FN_C"].append(FN_W[2])
            json_dict["Precision_A"].append(Precision_A)
            json_dict["Recall_A"].append(Recall_A)
            json_dict["F1_A"].append(F1_A)
            json_dict["Precision_B"].append(Precision_B)
            json_dict["Recall_B"].append(Recall_B)
            json_dict["F1_B"].append(F1_B)
            json_dict["Precision_C"].append(Precision_C)
            json_dict["Recall_C"].append(Recall_C)
            json_dict["F1_C"].append(F1_C)
            jfr.close()
        with open("log/{}.json".format(model_path), "w") as fout:
            json_str = json.dumps(json_dict, indent=2)
            fout.write(json_str)
            fout.close()

    def _get_metrics_ABC(self, True_P_W, False_P_W, False_N_W, item, mode):
        word_label_list = item["wordLabelList"]
        pred = item["pred"]
        tag = item["tag"]
        true_tor = self.config["WORD_TRUE_TOR"]
        false_tor = self.config["WORD_FALSE_TOR"]
        tokenized_sentence = self.tokenizer(item["input"], add_special_tokens=False, return_tensors="pt")
        word_ids = tokenized_sentence.word_ids()
        word_cnt = word_ids[-1]+1
        overlap_dict = self._overlap_dict.copy()
        user_index = item["userIndex"]
        data_index = item["dataIndex"]
        input_index = item["inputIndex"]
        len_bottom_line = len(self.data[item["dataIndex"]]["label"][item["inputIndex"][-1]])
        for word_id in range(word_cnt):
            a_word_TP = 0
            a_word_FP = 0
            a_word_FN = 0
            word_len = 0
            user_fix = None
            for index in range(len(word_ids)):
                weight_skip = True
                if(mode=="bottom" and index<(len(word_ids)-len_bottom_line)):
                    continue
                if(mode=="weight"):
                    for user in user_index:
                        head_cut = 0
                        tail_cut = 0
                        add_head = True
                        for index_in in input_index:
                            if(user==index_in):
                                add_head=False
                                continue
                            cut_len = len(self.data[data_index]["label"][index_in])
                            if add_head:
                                head_cut += cut_len 
                            else:
                                tail_cut += cut_len
                        head_index = head_cut
                        tail_index = len(word_ids) - tail_cut
                        if(index>=head_index and index<tail_index):
                            weight_skip = False
                            user_fix = user
                    if weight_skip:
                        continue

                if word_ids[index]!=word_id:
                    continue
                else:
                    word_len += 1
                    pred_label = pred[index]
                    tag_label = tag[index]
                    if(tag_label!='O' and tag_label==pred_label):
                        a_word_TP += 1
                    elif(pred_label!='O' and tag_label!=pred_label):
                        a_word_FP += 1
                    elif(pred_label=='O' and tag_label!='O'):
                        a_word_FN += 1
            if word_len==0:
                continue
            True_P_W_tmp = [0,0,0]
            False_P_W_tmp = [0,0,0]
            False_N_W_tmp = [0,0,0]
            # 全对才算
            if a_word_TP==word_len:
                True_P_W_tmp[0] += 1
            if a_word_FP>0:
                False_P_W_tmp[0] += 1
            if a_word_FN>0:
                False_N_W_tmp[0] +=1
            # 容错
            if a_word_TP/word_len>=true_tor:
                True_P_W_tmp[1] += 1
            if a_word_FP/word_len>=false_tor:
                False_P_W_tmp[1] += 1
            if a_word_FN/word_len>=false_tor:
                False_N_W_tmp[1] += 1
            # 一个就算对
            if a_word_TP>0:
                True_P_W_tmp[2] += 1
            if a_word_FP==word_len:
                False_P_W_tmp[2] += 1
            if a_word_FN==word_len:
                False_N_W_tmp[2] +=1
            if(mode=="weight"):
                weight = self._overlap_dict[item["dataIndex"]][user_fix]  
                for i in range(3):
                    True_P_W_tmp[i] = True_P_W_tmp[i]/weight
                    False_P_W_tmp[i] = False_P_W_tmp[i]/weight
                    False_N_W_tmp[i] = False_N_W_tmp[i]/weight
            for i in range(3):
                True_P_W[i] += True_P_W_tmp[i]
                False_P_W[i] += False_P_W_tmp[i]
                False_N_W[i] += False_N_W_tmp[i]
        return True_P_W, False_P_W, False_N_W

            
    def eval_all(self):
        True_P = 0
        False_P = 0
        False_N = 0
        True_P_W = [0, 0, 0]
        False_P_W = [0, 0, 0]
        False_N_W = [0, 0, 0]
        for item in self.inputs:
            per_y_true = item["tag"]
            per_y_pred = item["pred"]
            write_bad_case_flag = False
            for true_label, pred_label in zip(per_y_true, per_y_pred):
                if(true_label!='O' and true_label==pred_label):
                    True_P += 1
                elif(pred_label!='O' and true_label!=pred_label):
                    write_bad_case_flag = True
                    False_P += 1
                elif(pred_label=='O' and true_label!='O'):
                    write_bad_case_flag = True
                    False_N += 1
            True_P_W, False_P_W, False_N_W = self._get_metrics_ABC(True_P_W, False_P_W, False_N_W, item, "all")

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
        print("evaluation of all")
        print("The Window SIZE is:  "+str(self.config["SLIDING_WIN_SIZE"]))
        print("There are totally {} tags.".format(self.total_label))
        print("TF = "+str(True_P)+"\t\t"+"FP = "+str(False_P)+"\t\t"+"FN = "+str(False_N))
        Precision, Recall, F1 = self._get_metrics(True_P, False_P, False_N)
        print("The Precision is:{}".format(Precision), end="\t")
        print("The Recall is:{}".format(Recall), end="\t")
        print("The F1 is:{}".format(F1))
        print("Using time {} in prediction".format(self.predict_time))
        print("The word metrics:")
        print(True_P_W)
        print(False_P_W)
        print(False_N_W)
        print("**"*20)
        self.write_log("eval_all", True_P, False_P, False_N, True_P_W, False_P_W, False_N_W)

    def eval_bottom_line(self):
        True_P = 0
        False_P = 0
        False_N = 0
        True_P_W = [0, 0, 0]
        False_P_W = [0, 0, 0]
        False_N_W = [0, 0, 0]
        for item in self.inputs:
            bottom_index = item["inputIndex"][-1]
            len_label = len(self.data[item["dataIndex"]]["label"][bottom_index])
            per_y_true = item["tag"][-len_label:]
            per_y_pred = item["pred"][-len_label:]
            for true_label, pred_label in zip(per_y_true, per_y_pred):
                if(true_label!='O' and true_label==pred_label):
                    True_P += 1
                elif(pred_label!='O' and true_label!=pred_label):
                    False_P += 1
                elif(pred_label=='O' and true_label!='O'):
                    False_N += 1
            True_P_W, False_P_W, False_N_W = self._get_metrics_ABC(True_P_W, False_P_W, False_N_W, item, "bottom")
            
        print("model_path:"+self.config["MODEL_PATH"])
        print("evaluation of bottom line")
        print("**"*20)
        print("The Window SIZE is:  "+str(self.config["SLIDING_WIN_SIZE"]))
        print("There are totally {} tags.".format(self.total_label))
        print("TF = "+str(True_P)+"\t\t"+"FP = "+str(False_P)+"\t\t"+"FN = "+str(False_N))
        Precision, Recall, F1 = self._get_metrics(True_P, False_P, False_N)
        print("The Precision is:{}".format(Precision), end="\t")
        print("The Recall is:{}".format(Recall), end="\t")
        print("The F1 is:{}".format(F1))
        print("Using time {} in prediction".format(self.predict_time))
        print("The word metrinput")
        print(True_P_W)
        print(False_P_W)
        print(False_N_W)
        print("**"*20)
        self.write_log("eval_bottom", True_P, False_P, False_N, True_P_W, False_P_W, False_N_W)

    def eval_weighted_user(self):
        # 统计overlap次数，并且把user抽取出来
        True_P_W = [0, 0, 0]
        False_P_W = [0, 0, 0]
        False_N_W = [0, 0, 0]
        self._overlap_dict = {}
        for item in self.inputs:
            data_index = item["dataIndex"]
            if not data_index in self._overlap_dict.keys():
                self._overlap_dict[data_index] = {}
            user_index = item["userIndex"]
            input_index = item["inputIndex"]
            for index in user_index:
                if not index in self._overlap_dict[data_index].keys():
                    self._overlap_dict[data_index][index] = 1
                else:
                    self._overlap_dict[data_index][index] += 1
            True_P_W, False_P_W, False_N_W = self._get_metrics_ABC(True_P_W, False_P_W, False_N_W, item, "weight")
            all_tag = item["tag"]
            all_pred = item["pred"]
            user_tag_list = []
            user_pred_list = []
            for index in user_index:
                head_cut = 0
                tail_cut = 0
                add_head = True
                for index_in in input_index:
                    if(index==index_in):
                        add_head=False
                        continue
                    cut_len = len(self.data[data_index]["label"][index_in])
                    if add_head:
                        head_cut += cut_len 
                    else:
                        tail_cut += cut_len
                user_tag_list.append(all_tag[head_cut:(len(user_tag_list)-tail_cut-1)])
                user_pred_list.append(all_pred[head_cut:(len(user_pred_list)-tail_cut-1)])
            item["user_tag_list"] = user_tag_list
            item["user_pred_list"] = user_pred_list


        True_P = 0
        False_P = 0
        False_N = 0
        for item in self.inputs:
            data_index = item["dataIndex"]
            user_list = item["userIndex"]
            per_y_true = item["tag"]
            per_y_pred = item["pred"]
            for a_user_label, a_pred_label, user_index in zip(item["user_tag_list"], item["user_pred_list"], user_list):
                weight = self._overlap_dict[data_index][user_index]
                True_P_tmp = 0
                False_P_tmp = 0
                False_N_tmp = 0
                for true_label, pred_label in zip(a_user_label, a_pred_label):
                    if(true_label!='O' and true_label==pred_label):
                        True_P_tmp += 1
                    elif(pred_label!='O' and true_label!=pred_label):
                        False_P_tmp += 1
                    elif(pred_label=='O' and true_label!='O'):
                        False_N_tmp += 1
                True_P_tmp = True_P_tmp/weight
                False_P_tmp = False_P_tmp/weight
                False_N_tmp = False_N_tmp/weight
                True_P += True_P_tmp
                False_P += False_P_tmp
                False_N += False_N_tmp
        print("model_path:"+self.config["MODEL_PATH"])
        print("**"*20)
        print("evaluation of weighted user")
        print("The Window SIZE is:  "+str(self.config["SLIDING_WIN_SIZE"]))
        print("There are totally {} tags.".format(self.total_label))
        print("TF = "+str(True_P)+"\t\t"+"FP = "+str(False_P)+"\t\t"+"FN = "+str(False_N))
        Precision, Recall, F1 = self._get_metrics(True_P, False_P, False_N)
        print("The Precision is:{}".format(Precision), end="\t")
        print("The Recall is:{}".format(Recall), end="\t")
        print("The F1 is:{}".format(F1))
        print("Using time {} in prediction".format(self.predict_time))
        print("The word metrics:")
        print(True_P_W)
        print(False_P_W)
        print(False_N_W)
        print("**"*20)
        self.write_log("eval_weighted_user", True_P, False_P, False_N, True_P_W, False_P_W, False_N_W)
        self._overlap_dict = {}
        
        

    def write_badcase(self):
        model_path = "_".join(self.config["MODEL_PATH"].split("/"))
        win_size = self.config["SLIDING_WIN_SIZE"]
        input_type = self.inputs[0]["inputType"]
        if self.config["USE_SPECIAL_TOKENS"]:
            input_type = input_type + "_S"
        with open("badcases/{}_Win{}_{}.txt".format(model_path, win_size, input_type),  "w") as bf:
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
                  "xlm-roberta-large-finetuned-conll03-english",
                  "dslim/bert-base-NER",  
                  "dslim/bert-large-NER",
                  "vlan/bert-base-multilingual-cased-ner-hrl",
                  "dbmdz/bert-large-cased-finetuned-conll03-english",
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
            evaluator.eval_bottom_line()
            evaluator.eval_weighted_user()

            evaluator.collate_inputs_Only_User()
            evaluator.get_predict_label()
            evaluator.eval_all()
            evaluator.write_badcase()
            evaluator.eval_bottom_line()
            evaluator.eval_weighted_user()


if __name__=="__main__":
    main()

