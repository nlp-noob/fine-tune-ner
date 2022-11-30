import json
import os
import random

PRE_TAGGED_DATA = "eval_data/pretagged_empty_big_fixed.json"
FILTER_DIR = "train_data/per_filter_big/"
OUT_DIR = "train_data/per_big_new/"
P_VALID_TRAIN = 0.1
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
    order_valid_bad, order_train_bad = split_training_set(order_bad)
    order_valid.extend(order_valid_bad)    
    order_train.extend(order_train_bad)
     
    write_order(OUT_DIR+"train0000.json", order_train)
    write_order(OUT_DIR+"valid0000.json", order_valid)


def main():
    filter_all()


if __name__=="__main__":
    main()
