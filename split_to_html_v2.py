import json

HTML_DIR = "htmls/per/"
OUT_DIR = "data/"
WIN_SIZE = 2
ADD_INFO_SIZE = 2



def get_main_template():
    with open(HTML_DIR+"main_template.html", "r") as hf:
        main_template_html = hf.readlines()
        hf.close()
    return "".join(main_template_html)

def get_js_button():
    with open(HTML_DIR+"js_code_button.js", "r") as hf:
        js_code = hf.readlines()
        hf.close()
    return "".join(js_code)

def get_style_template():
    with open(HTML_DIR+"style_template.html", "r") as hf:
        main_template_html = hf.readlines()
        hf.close()
    return "".join(main_template_html)

def get_check_box_tail():
    with open(HTML_DIR+"get_check_box_tail.js", "r") as hf:
        js_code = hf.readlines()
        hf.close()
    return "".join(js_code)

def add_up_label_list(label_list):
    new_label_list = []
    for label in label_list:
        new_a_label = []
        for index in label:
            new_a_label.append(index+1)
        new_label_list.append(new_a_label)
    return new_label_list
            

def main():
    with open("eval_data/pretagged_empty_big.json", "r") as jf:
        orders = json.loads(jf.read())

    content_list = []
    a_page = []
    for order_index in range(len(orders)):
        label_list = orders[order_index]["label"]
        order_list = orders[order_index]["order"]

        input_indexs = []
        for index in range(len(label_list)):
            label = label_list[index]
            if not order_list[index][0]:
                continue
            else:
                if label:
                    input_indexs.append(index)

        for input_index in input_indexs:
            label = label_list[input_index]
            label = add_up_label_list(label)
            item_dict = {
                        "order_pre": [],
                        "order_check": [
                                        [
                                        "[USER]: "+order_list[input_index][1]
                                        ]
                                       ],
                        "label_pre": [],
                        "label_check": [
                                        label
                                       ],
                        "id": len(a_page) if len(a_page) < 100 else 0,
                        "isSelect": False,
                        "index_order": order_index,
                        "index_line": input_index
            }
            for i in range(WIN_SIZE-1):
                index_up = i+1
                now_index = input_index-index_up
                if now_index<0:
                    continue
                else:
                    name = "[USER]: " if order_list[now_index][0] else "[ADVISOR]: "
                    text = order_list[now_index][1]
                    new_line = [[name+text]]
                    new_line.extend(item_dict["order_check"])
                    item_dict["order_check"] = new_line
                    new_label = [add_up_label_list(label_list[now_index])]
                    new_label.extend(item_dict["label_check"])
                    item_dict["label_check"] = new_label
                    
            for i in range(ADD_INFO_SIZE):
                index_up = i 
                now_index = input_index-index_up-WIN_SIZE
                if now_index<0:
                    continue
                else:
                    name = "[USER]: " if order_list[now_index][0] else "[ADVISOR]: "
                    text = order_list[now_index][1]
                    new_line = [[name+text]]
                    new_line.extend(item_dict["order_pre"])
                    item_dict["order_pre"] = new_line
                    new_label = [add_up_label_list(label_list[now_index])]
                    new_label.extend(item_dict["label_pre"])
                    item_dict["label_pre"] = new_label
            if len(a_page)>=100:
                content_list.append(a_page)
                a_page = [item_dict]
            else:
                a_page.append(item_dict)
    if a_page:
        content_list.append(a_page)
    style_template = get_style_template()
    main_template = get_main_template()
    js_code_button = get_js_button()
    check_box_tail = get_check_box_tail()
    for content_index in range(len(content_list)):
        if content_index < len(content_list)-1:
            next_page_name = "tagged_page"+f"{content_index+1:04d}"+".html"
        else:
            next_page_name = "thanks_page.html"
        this_page_name = "tagged_page"+f"{content_index:04d}"+".html"
        const_data = json.dumps(content_list[content_index], indent=2)
        title_str = "data" + str(content_index+1)
        html_page = main_template.format(style_template, 
                                         title_str,
                                         js_code_button,
                                         const_data,
                                         check_box_tail,
                                         "check"+str(content_index+1)+".txt",
                                         next_page_name)
        with open(HTML_DIR+OUT_DIR+this_page_name, "w") as fout:
            fout.write(html_page)
            fout.close()


if __name__=="__main__":
    main()

