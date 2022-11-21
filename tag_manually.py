import json
import readline

DATA_PATH = "eval_data/birth_untagged_data_small.json"

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
    for item in json_dict:
        orders = item["order"]
        labels = item["label"]
        new_label = []
        for order, label in zip(orders, labels):
            while(True):
                number_words_in_sentence(order[1])
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
                        new_label.append([])
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
        item["label"] = new_label
    jf.close()
    with open(DATA_PATH[:-5]+"_tagged"+".json", "w") as fout:
        json_str = json.dumps(json_dict, indent=2)
        fout.write(json_str)
        fout.close()
        print("Write Succes")


    

if __name__=="__main__":
    main()
