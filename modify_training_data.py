import json
import os
import random

PRE_TAGGED_DATA = "eval_data/per_data_big.json"
BAD_DATA = "train_data/per_big/bad_1_fixed.json"
FILTER_DIR = "train_data/per_filter_big/"
OUT_DIR = "train_data/per_big/"
P_VALID_TRAIN = 0.2
WRITE_MODE = "w"

def split_training_set(order_list):
    random.shuffle(order_list)
    len_valid = int(P_VALID_TRAIN*len(order_list))
    valid_data = order_list[:len_valid]
    train_data = order_list[len_valid:]
    return valid_data, train_data
    
def write_order(path, order_list):
    json_str = json.dumps(order_list, indent=2)
    if WRITE_MODE == "w":
        with open(path, WRITE_MODE) as fout:
            fout.write(json_str)
            fout.close()
    

# 错一个整个order不要
def filter_all():
    with open(PRE_TAGGED_DATA, "r") as originf:
        origin_data = json.loads(originf.read())
        originf.close()

    filter_list = []
    for filter_path in os.listdir(FILTER_DIR):
        a_filter = open(FILTER_DIR+filter_path,"r").readlines()
        for f in a_filter:
            order_index = int(f.split()[0])
            if not order_index in filter_list:
                filter_list.append(order_index)
    order_bad = []
    order_data = []
    for order_index in range(len(origin_data)):
        if order_index in filter_list:
            order_bad.append(origin_data[order_index])
        else:
            order_data.append(origin_data[order_index]) 
    order_valid, order_train = split_training_set(order_data)
    write_order(OUT_DIR+"train.json", order_train)
    write_order(OUT_DIR+"valid.json", order_valid)
    write_order(OUT_DIR+"bad.json", order_bad)

def _write_in_data(type_list, order_dict):
    tf = open(OUT_DIR+"train.json", "r")
    vf = open(OUT_DIR+"valid.json", "r")
    train_data = json.loads(tf.read())
    valid_data = json.loads(vf.read())
    tf.close()
    vf.close()
    small_list = []
    for a_type in type_list:
        order_list = order_dict[a_type]
        if len(order_list)<10:
            small_list.extend(order_list)
            continue
        else:
            random.shuffle(order_list)
            split_len = int(P_VALID_TRAIN*len(order_list))
            valid_part = order_list[split_len:]
            train_part = order_list[:split_len]
            train_data.extend(train_part)
            valid_data.extend(valid_part)
    if(len(small_list)>=10):
        split_len = int(P_VALID_TRAIN*len(small_list))
        valid_part = small_list[split_len:]
        train_part = small_list[:split_len]
        train_data.extend(train_part)
        valid_data.extend(valid_part)
    elif(len(small_list)>=2):
        split_len = 1
        valid_part = small_list[split_len:]
        train_part = small_list[:split_len]
        train_data.extend(train_part)
        valid_data.extend(valid_part)
    else:
        valid_data.extend(small_list)
    tfout = open(OUT_DIR+"train0001.json", "w")
    vfout = open(OUT_DIR+"valid0001.json", "w")
    train_json = json.dumps(train_data, indent=2)
    valid_json = json.dumps(valid_data, indent=2)
    tfout.write(train_json)
    vfout.write(valid_json)

def distribute_bad_data():
    jf = open(BAD_DATA, "r")
    bad_data = json.loads(jf.read())
    jf.close()
    type_list = []
    type_num = {}
    order_dict = {}
    for order in bad_data:
        if not order["type"] in type_list:
            type_list.append(order["type"])
            type_num[order["type"]] = 0
            order_dict[order["type"]] =[order]
        else:
            type_num[order["type"]] += 1
            order_dict[order["type"]].append(order)
    _write_in_data(type_list, order_dict)


# 分成部分的数据集
def filter_part():
    pass

def main():
    # filter_all()
    distribute_bad_data()


if __name__=="__main__":
    main()
