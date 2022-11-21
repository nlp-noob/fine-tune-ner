import re
import sys
import json

catch_text_compile = re.compile(r".*(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2}).(\d+).*")
pair_line = "==================== pair:"
order_line = "-------------------- order:"
new_line_symbol = " "
order_price_line = "\t order_price"
generate_empty_label = False


def get_labels_in_line(label_list, line_index):
    label_in_line = []
    for label in label_list:
        if label[0]-1 == line_index:
            label_in_line.append(label[1:])
    return label_in_line

def split_alnum_word(word):
    pieces_list = []
    a_piece = ""
    special_piece = ""
    for a_char in word:
        if a_char.isalpha():
            if special_piece:
                pieces_list.append(special_piece)
                special_piece = ""
            a_piece += a_char
        else:
            if a_piece:
                pieces_list.append(a_piece)
                a_piece = ""
            special_piece += a_char
    if a_piece:
        pieces_list.append(a_piece)
    if special_piece:
        pieces_list.append(special_piece)
    return pieces_list

def split_special_word(word):
    pieces_list = []
    a_piece = ""
    special_piece = ""
    for a_char in word:
        if a_char.isalnum():
            if special_piece:
                pieces_list.append(special_piece)
                special_piece = ""
            a_piece += a_char
        else:
            if a_piece:
                pieces_list.append(a_piece)
                a_piece = ""
            special_piece += a_char
    if a_piece:
        pieces_list.append(a_piece)
    if special_piece:
        pieces_list.append(special_piece)
    return pieces_list
    

def format_text(text):
    split_text = text.split()

    word_list = []
    for word in split_text:
        # 去除重复空格
        word = word.strip()
        # 字母数字组合才能作为一个词
        if word.isalnum():
            if word.isalpha():
                word_list.append(word)
            elif word.isdigit():
                word_list.append(word)
            else:
                word_list.extend(split_alnum_word(word))
                
        else:
            splitted_word_list = split_special_word(word)
            for splitted_word in splitted_word_list:
                if splitted_word.isalnum():
                    word_list.extend(split_alnum_word(splitted_word))
                else:
                    word_list.append(splitted_word)
    return " ".join(word_list)


def get_data(fin, label_list, name_list, byte_name_list):
    pair_no = -1
    orders = []
    order_no = 0
    an_order = {"pairNO":0,
                "orderNO":order_no,
                "order":[],
                "label":[] if generate_empty_label else label_list[len(orders)]}
    for line in fin:
    
        if not line.strip():
            continue

        if order_line in line and an_order["order"]:
            orders.append(an_order)
            order_no += 1
            an_order = {"pairNO":pair_no,
                        "orderNO":order_no,
                        "order":[],
                        "label":[] if generate_empty_label else label_list[len(orders)]}

        if pair_line in line:
            pair_no += 1

        if pair_line in line and an_order["order"]:
            orders.append(an_order)
            an_order = {"pairNO":pair_no,
                        "orderNO":order_no,
                        "order":[],
                        "label":[] if generate_empty_label else label_list[len(orders)]}
            order_no = 0

        if catch_text_compile.match(line):
            sentence = re.findall("\t.*\t(.*)\t:(.*)", line)
            name = sentence[0][0]
            if name.encode() in byte_name_list or name in name_list:
                name = False
            else:
                name = True
            text = sentence[0][1]
            text = format_text(text)

            an_order["order"].append([name,text])

        if not(pair_line in line 
               or catch_text_compile.match(line) 
               or order_line in line
               or order_price_line in line):
            text = format_text(line)
            an_order["order"][-1][1] += (new_line_symbol+text)

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

    if not generate_empty_label:
        with open("eval_data/label_birth.json", "r") as lf:
            label_list = json.load(lf)
    else:
        label_list = []

    with open(txt_file, 'r') as fin:
        orders = get_data(fin, label_list, name_list, byte_name_list)

    json_str = json.dumps(orders, indent=2)
    with open("eval_data/birth_data_small_test.json", "w") as jf: 
        jf.write(json_str)
        print("Write successed.")
    

if __name__=="__main__":
    main()

