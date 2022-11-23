import json
import os
import random

PRE_TAGGED_DATA = "eval_data/per_data_big.json"
FILTER_DIR = "train_data/per_filter_big/"
OUT_DIR = "train_data/per_big/"
P_TEST_TRAIN = 0.1
WRITE_MODE = "w"

def split_training_set(order_list):
    random.shuffle(order_list)
    len_test = int(P_TEST_TRAIN*len(order_list))
    test_data = order_list[:2*len_test]
    train_data = order_list[2*len_test:]
    valid_data = test_data[:len_test]
    test_data = test_data[len_test:]
    return test_data, valid_data, train_data
    
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
    order_test, order_valid, order_train = split_training_set(order_data)
    import pdb; pdb.set_trace()
    write_order(OUT_DIR+"train.json", order_train)
    write_order(OUT_DIR+"valid.json", order_valid)
    write_order(OUT_DIR+"test.json", order_test)
    write_order(OUT_DIR+"bad.json", order_bad)



# 分成部分的数据集
def filter_part():
    pass

def main():
    filter_all()


if __name__=="__main__":
    main()
