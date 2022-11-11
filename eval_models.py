import os
import re
import time
import torch
import json
from transformers import AutoTokenizer, AutoModelForTokenClassification
from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score
from seqeval.metrics import classification_report

# 是否把sub-word全部标记的选项
LABEL_ALL_TOKENS = True

# 设置打标签的名字，分别对应前面的标签和后面的标签
PREVIOUS_LABEL = "B-PER"
CENTER_LABEL = "I-PER"
ALL_LABEL = "PER"

# 设置是否有前缀：
HAVE_PRE = True

# 设置窗口大小(决定实时识别)
MAX_SLIDING_WIN_SIZE = 10

# 检查模型label的列表
CHECK_LABEL_LIST = []


def get_data_from_json():
    file_name = "eval_data/eval_data.json"
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
    return label_list, conversation_list


def check_the_label(predicted_token_classes):
    for pred_class in predicted_token_classes:
        if pred_class!='O' and pred_class not in CHECK_LABEL_LIST:
            CHECK_LABEL_LIST.append(pred_class)


def get_predict_label(conversation, model, tokenizer, add_special_tokens):
    # 这里的add_special_tokens要根据模型而定，通过之前的测试，这个模型添加这个得到的推导结果比不添加好得多。 
    inputs = tokenizer(conversation, add_special_tokens=add_special_tokens, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_token_class_ids = logits.argmax(-1)
    predicted_tokens_classes = [model.config.id2label[t.item()] for t in predicted_token_class_ids[0]]
    check_the_label(predicted_tokens_classes)
    if(add_special_tokens):
        del(predicted_tokens_classes[-1])
        del(predicted_tokens_classes[0])
    #########################
    # 去掉与本次任务无关的标签
    # 输出一下任务的标签
    not_O_is_print = False
    for pred_label_index in range(len(predicted_tokens_classes)):
        if(predicted_tokens_classes[pred_label_index]!="O" and not_O_is_print):
            print(predicted_tokens_classes[pred_label_index])
            not_O_is_print = True
        if(predicted_tokens_classes[pred_label_index]!=PREVIOUS_LABEL and
           predicted_tokens_classes[pred_label_index]!=CENTER_LABEL and
           predicted_tokens_classes[pred_label_index]!=ALL_LABEL):
            predicted_tokens_classes[pred_label_index]="O"
    #########################
    return predicted_tokens_classes


def align_label(sentence, label, tokenizer):
    # 这里传入的label就是当前这句话中所有标记的坐标
    # 如果label为空表示这里面没有标注
    joined_sentence = " ".join(sentence)
    inputs = tokenizer(joined_sentence, add_special_tokens=False, return_tensors="pt")
    word_ids = inputs.word_ids()
    tokenized_sentence = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    # 计算坐标的偏移
    label_bias = 0
    for token_id in range(len(tokenized_sentence)):
        if tokenized_sentence[token_id]==None:
            continue 
        if ":" in tokenized_sentence[token_id]:
            label_bias = word_ids[token_id] + 1
            break
    aligned_label = []
    for token_ids in range(len(tokenized_sentence)):
        aligned_label.append("O")
    for a_label in label:
        for label_index in range(len(a_label)):
            if(label_index==0):
                label_name = PREVIOUS_LABEL if HAVE_PRE else ALL_LABEL
            else:
                label_name = CENTER_LABEL if HAVE_PRE else ALL_LABEL
            for label_to_modify_index in range(len(aligned_label)):
                if word_ids[label_to_modify_index]==a_label[label_index]+label_bias:
                   aligned_label[label_to_modify_index]=label_name
    return aligned_label


# 用来产生文本形式的输入输出的函数
def modify_inputs_and_labels_text1(conversation_list, label_list, tokenizer):
    inputs = []
    labels = []
    for i in range(len(conversation_list)):
        conversation = conversation_list[i]
        label = label_list[i]
        user_indexs = []
        aligned_labels = []
        #############################
        # 将label转换成tokenized的形式
        for index in range(len(conversation)):
            label_to_convert = []
            for one_label in label:
                if(len(one_label)>0 and one_label[0]==(index+1)):
                    a_label_in_sentence = []
                    for label_index in range(len(one_label)):
                        if label_index==0:
                            continue
                        else:
                            a_label_in_sentence.append(one_label[label_index])
                    label_to_convert.append(a_label_in_sentence)
            aligned_labels.append(align_label(conversation[index],label_to_convert,tokenizer))    
        # 将label转换成tokenized的形式
        ##############################
        ##############################
        # 按照我们需要的形式对文本进行组合，同时将label用同样的方式组合在一起
        for index in range(len(conversation)):
            if(conversation[index][0]=="[USER]"):
                user_indexs.append(index)
        for index in user_indexs:
            if(index==0 and len(conversation)>1):
                a_sen = conversation[index].copy()
                b_sen = conversation[index+1].copy()
                a_sen.extend(b_sen)
                sen = a_sen
                a_label = aligned_labels[index].copy()
                b_label = aligned_labels[index+1].copy()
                a_label.extend(b_label)
                sen_label = a_label
            elif(index==0 and len(conversation)==1):
                sen = conversation[index]
                sen_label = aligned_labels[index]
            elif(index==len(conversation)-1):
                a_sen = conversation[index-1].copy()
                b_sen = conversation[index].copy()
                a_sen.extend(b_sen)
                sen = a_sen
                a_label = aligned_labels[index-1].copy()
                b_label = aligned_labels[index].copy()
                a_label.extend(b_label)
                sen_label = a_label
            else:
                a_sen = conversation[index-1].copy()
                b_sen = conversation[index].copy()
                c_sen = conversation[index+1].copy()
                a_sen.extend(b_sen)
                a_sen.extend(c_sen)
                sen = a_sen 
                a_label = aligned_labels[index-1].copy()
                b_label = aligned_labels[index].copy()
                c_label = aligned_labels[index+1].copy()
                a_label.extend(b_label)
                a_label.extend(c_label)
                sen_label = a_label
            symbol = ' ' 
            sen = symbol.join(sen)
            inputs.append(sen)
            labels.append(sen_label)
    return inputs, labels


# 用来产生文本形式的输入输出的函数
def modify_inputs_and_labels_text1_WIN(conversation_list, label_list, tokenizer, sliding_win_size):
    inputs = []
    labels = []
    for i in range(len(conversation_list)):
        conversation = conversation_list[i]
        label = label_list[i]
        user_indexs = []
        aligned_labels = []
        #############################
        # 将label转换成tokenized的形式
        for index in range(len(conversation)):
            label_to_convert = []
            for one_label in label:
                if(len(one_label)>0 and one_label[0]==(index+1)):
                    a_label_in_sentence = []
                    for label_index in range(len(one_label)):
                        if label_index==0:
                            continue
                        else:
                            a_label_in_sentence.append(one_label[label_index])
                    label_to_convert.append(a_label_in_sentence)
            aligned_labels.append(align_label(conversation[index],label_to_convert,tokenizer))    
        # 将label转换成tokenized的形式
        ##############################
        ##############################
        # 按照我们需要的形式对文本进行组合，同时将label用同样的方式组合在一起
        # 这里把所有的user的坐标记录下来
        for index in range(len(conversation)):
            if(conversation[index][0]=="[USER]"):
                user_indexs.append(index)
        for index in user_indexs:
            sen = conversation[index].copy()
            sen_label = aligned_labels[index].copy()
            for up_index in range(sliding_win_size+1):
                if((index-up_index-1)>=0):
                    sen = conversation[index-up_index-1] + sen
                    sen_label = aligned_labels[index-up_index-1] + sen_label
            symbol = ' '
            sen = symbol.join(sen)
            inputs.append(sen)
            labels.append(sen_label)
    return inputs, labels


def get_eval(per_y_true, per_y_pred):
    print("len per_y_true: "+str(len(per_y_true)))
    print("len per_y_pred: "+str(len(per_y_pred)))
    print("**"*20)
    True_P = 0
    False_P = 0
    False_N = 0
    for true_label, pred_label in zip(per_y_true, per_y_pred):
        if(true_label!='O' and true_label==pred_label):
            True_P += 1
        elif(pred_label!='O' and true_label!=pred_label):
            False_P += 1
        elif(pred_label=='O' and true_label!='O'):
            False_N += 1
    print(True_P)
    print(False_P)
    print(False_N)
    Precision = True_P/(True_P + False_P)
    Recall = True_P/(True_P + False_N)
    F1 = 2*Precision*Recall/(Precision+Recall)
    print("The Precision is:\t {}".format(Precision))
    print("The Recall is:\t\t {}".format(Recall))
    print("The F1 is:\t\t {}".format(F1))
    pass


def main():
    labels, conversations = get_data_from_json()
    print(labels)
    model_path = "Davlan/bert-base-multilingual-cased-ner-hrl"
    print("the model is: {}".format(model_path))
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)

    print("=="*20)
    print("这一部分是基于后期识别的(非实时)")
    print(labels)
    print("There are {} conversations".format(len(conversations)))
    print("There are {} labels".format(len(labels)))
    inputs, true_labels = modify_inputs_and_labels_text1(conversations,labels,tokenizer)
    per_y_true = []
    per_y_pred = []
    start_time = time.time()
    for i in range(len(inputs)): 
        predict_labels = get_predict_label(inputs[i], model, tokenizer, add_special_tokens=True)
        per_y_true.extend(true_labels[i])
        per_y_pred.extend(predict_labels)

    # 输出一下label检查设置的有没有出错：
    print("**"*20)
    print("The Check Label List is {}".format(CHECK_LABEL_LIST))
    print("**"*20)
    get_eval(per_y_true, per_y_pred)
    end_time = time.time()
    print("Using time: {}".format(end_time-start_time), 's')
    print("=="*20)

    print("=="*20)
    print("这一部分是基于滑动窗口进行评估的(实时)")
    for sliding_win_size in range(MAX_SLIDING_WIN_SIZE+1):
        start_time = time.time()
        print("**"*20)
        print("The Sliding Window size is: {}".format(sliding_win_size+1))
        print("**"*20)
        inputs, true_labels = modify_inputs_and_labels_text1_WIN(conversations,labels,tokenizer, sliding_win_size+1)
        per_y_true = []
        per_y_pred = []
        for i in range(len(inputs)): 
            predict_labels = get_predict_label(inputs[i], model, tokenizer, add_special_tokens=True)
            per_y_true.extend(true_labels[i])
            per_y_pred.extend(predict_labels)
        get_eval(per_y_true, per_y_pred)
        end_time = time.time()
        print("Using time: {}".format(end_time-start_time), 's')
        print("=="*20)


if __name__=="__main__":
    main()
