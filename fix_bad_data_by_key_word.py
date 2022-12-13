import json
import readline


DATA_PATH = "train_data/per_big_new/valid0001_01.json"
OUTPUT_PATH = "train_data/per_big_new/valid0001_01_fixxed.json"
BADCASE_FILE = "badcases_train/dslim_bert-large-NER_win3_cosine/bad_case_for_step_00002400.txt"
FORBIDDEN_WORDS = ["er", "ke"]


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

def search_words(data_to_fix, words):
    if len(words)==1 and len(words[0])<=1:
        return []
    elif len(words)==1 and words[0] in FORBIDDEN_WORDS:
        return []
    index_list = []
    for order_index, order in enumerate(data_to_fix):
        for line_index, line in enumerate(order["order"]):
            if not line[0]:
                continue
            # words全部都在这行才算
            output_index_flag = True
            for a_word in words:
                if not a_word in line[1]:
                    output_index_flag = False
            if output_index_flag:
                index_list.append([order_index,line_index])
    if len(index_list)>10:
        print(words)
        raise ValueError("There too manny results for tagging plz check if the words above is too short") 
    if len(index_list)==0:
        return False 
    else:
        return index_list

     
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


def main():
    with open(DATA_PATH, "r") as jf:
        data_to_fix = json.load(jf)
        jf.close()

    # get bad words:
    bad_words_line_indexs = []
    all_words_pattern = "all_words:\t" 
    with open(BADCASE_FILE, "r") as bf:
        bad_data = bf.readlines()
        for order_index, line in enumerate(bad_data):
            if all_words_pattern in line:
                all_words = line.strip().split("\t")[1:]
                line_index = search_words(data_to_fix, all_words)
                if line_index:
                    bad_words_line_indexs.extend(line_index)
                elif line_index!=False:
                    continue
                else:
                    print(all_words)
                    raise ValueError("There is not corresponding wrong words above in the data, please check it manually!!!") 

    quit_flag = False
    # show and fix the bad word index
    for bad_fix_progress, bad_word_index in enumerate(bad_words_line_indexs):
        order_index = bad_word_index[0]
        line_index = bad_word_index[1]
        label = data_to_fix[order_index]["label"][line_index]
        label_fix = []
        while(True):
            print("=="*30)
            print("processing bad_data \t{}/{}".format(bad_fix_progress+1, len(bad_words_line_indexs)))
            print("=="*30)
            if line[0]:
                print("[USER]:")
            else:
                print("[ADVISOR]:")
            number_words_in_sentence(data_to_fix[order_index]["order"][line_index][1])
            print(label)
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
                        label_fix = []
                        print("=="*20)
                        print("You insert an empty label list")
                        print("=="*20)
                        break
                    elif(check_right_list(enter_list)):
                        label_fix = enter_list
                        print("=="*20)
                        print("You insert a list of label:")
                        print(enter_list)
                        print("=="*20)
                        break
                if new_label_list == "j":
                    label_fix = label
                    break
                elif new_label_list == "quit":
                    quit_flag = True    
                    break
                else:
                    print("**"*20)
                    print("Wrong input!!!")
                    print("you should enter a list of label or just enter \"j\" to jump(empty label)")
                    print("To quit and save, enter: \"quit\"")
                    print("**"*20)
            if quit_flag:
                break
            
        if quit_flag:
            break
        else:
            data_to_fix[order_index]["label"][line_index] = label_fix
    json_str = json.dumps(data_to_fix, indent=2)
    with open(OUTPUT_PATH, "w") as fout:
        fout.write(json_str)
        print("File has been saved to the path: {}".format(OUTPUT_PATH))
        fout.close()
        

if __name__=="__main__":
    main()

