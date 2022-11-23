import json
import readline

DATA_PATH = "train_data/per_big/bad_1.json"
continue_process_flag = True
save_every_order = True

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


def main():
    command_list = ["j"]
    with open(DATA_PATH, "r") as jf:
        json_dict = json.load(jf)
    quit_flag = False

    # 用来存储所有类别的字典
    type_info = {}
    type_list = []

    for item, progress_order in zip(json_dict, range(len(json_dict))):
        if len(item)==6 and continue_process_flag:
            if not item["type"] in type_list:                
                type_list.append(item["type"])
                type_info[item["type"]] = item["type_info"]
            continue

        order = item["order"]
        labels = item["label"]
        new_label = []
        for line, label, line_index in zip(order, labels, range(len(order))):
            while(True):
                print("=="*30)
                print("processing order \t{}/{}".format(progress_order, len(json_dict)))
                print("processing line \t{}/{}".format(line_index, len(order)))
                print("=="*30)
                if line[0]:
                    print("[USER]:")
                else:
                    print("[ADVISOR]:")
                number_words_in_sentence(line[1])
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
                            new_label.append([])
                            print("=="*20)
                            print("You insert an empty label list")
                            print("=="*20)
                            break
                        elif(check_right_list(enter_list)):
                            new_label.append(enter_list)
                            print("=="*20)
                            print("You insert a list of label:")
                            print(enter_list)
                            print("=="*20)
                            break
                        
                    if new_label_list == "j":
                        new_label.append(label)
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

        # 进行类别的输入
        while(True):
            if not type_info:
                print("\"\""*20)
                print("There is no type in the list!")
                print("\"\""*20)
            else:
                print("\"\""*20)
                for a_type_index in range(len(type_list)):
                    a_type = type_list[a_type_index]
                    print("{}:\t {}\t\t type info is: {}".format(a_type_index, a_type, type_info[a_type]))
                print("\"\""*20)

            type_input = input("please enter the type of the order:")
            if not type_input:
                print(empty_input)
                continue
            if type_input.isdigit():
                a_type_index = int(type_input)
                if a_type_index >= len(type_list) or a_type_index < 0:
                    print("out of range!!")
                    continue
                else:
                    type_input = type_list[a_type_index]
                    print("\"\""*20)
                    print("choosing type {}".format(type_input))
            item["type"] = type_input
            if not type_input in type_list:
                type_text = input("please enter the type info of this order:")
                type_info[type_input] = type_text
                type_list.append(type_input)
            item["type_info"] = type_info[type_input]
            break
        item["label"] = new_label

        # 随时保存
        if save_every_order:
            with open(DATA_PATH[:-5]+"_fixed"+".json", "w") as fout:
                json_str = json.dumps(json_dict, indent=2)
                fout.write(json_str)
                fout.close()
                print("Write Succes")
            with open(DATA_PATH[:-5]+"_bak"+".json", "w") as fout:
                json_str = json.dumps(json_dict, indent=2)
                fout.write(json_str)
                fout.close()
                print("Write Succes")

    jf.close()
    with open(DATA_PATH[:-5]+"_fixed"+".json", "w") as fout:
        json_str = json.dumps(json_dict, indent=2)
        fout.write(json_str)
        fout.close()
        print("Write Succes")


    

if __name__=="__main__":
    main()
