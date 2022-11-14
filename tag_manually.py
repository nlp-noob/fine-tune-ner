import json
import readline

DATA_PATH = "eval_data/eval_data.json"

def main():
    with open(DATA_PATH, "r") as jf:
        json_dict = json.load(jf)
    for item in json_dict:
        orders = item["order"]
        labels = item["label"]
        new_label = []
        for order, label in zip(orders, labels):
            print(order)
            print(label)
            new_label_list = input("please input the new labels:")
            print(type(new_label_list))

            
    

if __name__=="__main__":
    main()
