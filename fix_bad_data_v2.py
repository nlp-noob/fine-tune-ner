import os
import json
import readline

CHECK_WIN_SIZE = 3
FILTER_DIR = "train_data/per_filter_big/"
DATA_PATH = "eval_data/pretagged_empty_big.json"
save_every_order = True

def get_filter_list():
    filter_list = []
    for filter_path in os.listdir(FILTER_DIR):
        a_filter = open(FILTER_DIR+filter_path,"r").readlines()
        for f in a_filter:
            order_index = int(f.split()[0])
            line_index  = int(f.split()[1])
            if not order_index in filter_list:
                filter_list.append([order_index, line_index])
    return filter_list

def find_index(filter_list, index):
    index_is_bad = False
    for bad_index, index_filter in zip(filter_list, range(len(filter_list))):
        if index[0]==bad_index[0] and index[1]==bad_index[1]:
            index_is_bad = index_filter
    return index_is_bad


def check_right_list(check_list):
    for item in check_list:
        if(str(type(item))!="<class 'list'>"):
            return False
        elif(len(item)==0):
            return False
        for sub_item in item:
            if(str(type(sub_item))!="<class 'int'>"):
                return False
    return True

def number_words_in_sentence(sentence):
    word_list = sentence.split()
    words_str = ""
    index_str = ""
    for index in range(len(word_list)):
        space_bias = abs(len(word_list[index])-len(str(index)))
        
        if(len(word_list[index])>len(str(index))):

            words_str = words_str + word_list[index] + "\t" 
            index_str = index_str + str(index) + " "*space_bias + "\t" 
        else:
            words_str = words_str + word_list[index] + " "*space_bias + "\t"
            index_str = index_str + str(index) + "\t" 
            
    print(words_str)
    print(index_str)

def get_input_label(display_line, display_bad, display_label, filter_list):
    return_label = []
    quit_flag = False
    while(True):
        print("=="*30)
        print("processing bad \t{}/{}".format(display_bad, len(filter_list)))
        print("=="*30)
        if display_line[0]:
            print("[USER]:")
        else:
            print("[ADVISOR]:")
        number_words_in_sentence(display_line[1])
        print(display_label)
        print("label the PER in the sentence like this: [[0,1],[4,5]]")
        new_label_list = input("please input the new labels:") 
        enter_list = None
        try:
            enter_list = json.loads(new_label_list)
        except:
            print("wrong json typr")
        finally:
            if str(type(enter_list))=="<class 'list'>":
                if(len(enter_list)==0):
                    print("=="*20)
                    print("You insert an empty label list")
                    print("=="*20)
                    break
                elif(check_right_list(enter_list)):
                    return_label = enter_list
                    print("=="*20)
                    print("You insert a list of label:")
                    print(enter_list)
                    print("=="*20)
                    break
            if new_label_list == "j":
                return display_label, quit_flag
            elif new_label_list == "quit":
                quit_flag = True    
                break
            else:
                print("**"*20)
                print("Wrong input!!!")
                print("you should enter a list of label or just enter \"j\" to jump")
                print("To quit and save, enter: \"quit\"")
                print("**"*20)
    return return_label, quit_flag


def main():
    with open(DATA_PATH, "r") as jf:
        json_dict = json.load(jf)
        jf.close()
    filter_list = get_filter_list()
    quit_flag = False
    processing_cnt = 0

    for item, order_index in zip(json_dict, range(len(json_dict))):

        order = item["order"]
        labels = item["label"]
        for line, label, line_index in zip(order, labels, range(len(order))):
            bad_process = find_index(filter_list, [order_index, line_index])
            if not bad_process:
                continue
            else:
                processing_cnt += 1
                return_label, quit_flag = get_input_label(line, processing_cnt, label, filter_list) 
                item["label"][line_index] = return_label
                for up_index in range(CHECK_WIN_SIZE):
                    now_index = line_index - up_index - 1
                    if now_index >=0:
                        line_now = item["order"][now_index]
                        label_now = item["label"][now_index]
                        return_label, quit_flag = get_input_label(line_now, processing_cnt, label_now, filter_list) 
                        item["label"][now_index] = return_label
                        if quit_flag:
                            break
        if quit_flag:
            break

        # 随时保存
        if save_every_order:
            with open(DATA_PATH[:-5]+"_fixed"+".json", "w") as fout:
                json_str = json.dumps(json_dict, indent=2)
                fout.write(json_str)
                fout.close()
            with open(DATA_PATH[:-5]+"_bak"+".json", "w") as fout:
                json_str = json.dumps(json_dict, indent=2)
                fout.write(json_str)
                fout.close()

    with open(DATA_PATH[:-5]+"_fixed"+".json", "w") as fout:
        json_str = json.dumps(json_dict, indent=2)
        fout.write(json_str)
        fout.close()
        print("Write Succes")


    

if __name__=="__main__":
    main()
