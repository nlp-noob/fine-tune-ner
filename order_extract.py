import re
import sys
import json

catch_text_compile = re.compile(r".*(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2}).(\d+).*")
pair_line = "==================== pair:"
order_line = "-------------------- order:"

def get_labels_in_line(label_list, line_index):
    label_in_line = []
    for label in label_list:
        if label[0]-1 == line_index:
            label_in_line.append(label[1:])
    return label_in_line

def get_data(fin, label_list, name_list, byte_name_list):
    pair_no = -1
    orders = []
    order_no = 0
    an_order = {"pairNO":pair_no,
                "orderNO":order_no,
                "order":[],
                "label":label_list[len(orders)]}
    for line in fin:
    
        if not line.strip():
            continue

        if order_line in line and an_order["order"]:
            orders.append(an_order)
            order_no += 1
            an_order = {"pairNO":pair_no,
                        "orderNO":order_no,
                        "order":[],
                        "label":label_list[len(orders)]}

        if pair_line in line:
            pair_no += 1

        if pair_line in line and an_order["order"]:
            orders.append(an_order)
            an_order = {"pairNO":pair_no,
                        "orderNO":order_no,
                        "order":[],
                        "label":label_list[len(orders)]}
            order_no = 0

        if catch_text_compile.match(line):
            sentence = re.findall("\t.*\t(.*)\t:(.*)", line)
            name = sentence[0][0]
            if name.encode() in byte_name_list or name in name_list:
                name = "[ADVISOR]"
            else:
                name = "[USER]"
            text = sentence[0][1].strip()
            an_order["order"].append([name,text])

    orders.append(an_order)

    # 把label转换成每行一个
    for order in orders:
        label_list = order["label"]
        order["label"] = []
        for index in range(len(order["order"])):
            order["label"].append(get_labels_in_line(label_list, index))
            
    return orders


def main():

    txt_file = sys.argv[1] if len(sys.argv) > 1 else 'eval_data/order.sample.txt'
    
    with open("eval_data/advisor_name_byte.json", "r") as bnf:
        name_list = json.load(bnf)
        byte_name_list = []
        for name in name_list:
            byte_name_list.append(name.encode())

    with open("eval_data/label.json", "r") as lf:
        label_list = json.load(lf)

    with open(txt_file, 'r') as fin:
        orders = get_data(fin, label_list, name_list, byte_name_list)

    json_str = json.dumps(orders, indent=2)
    with open("eval_data/eval_data_test.json", "w") as jf: 
        jf.write(json_str)
        print("Write successed.")
    

if __name__=="__main__":
    main()







