import json

HTML_DIR = "htmls/"

def mark_words(text, labels):
    words = text.split()
    flatten_label = []
    for label in labels:
        words[label[0]] = "<mark>" + words[label[0]]
        words[label[-1]] = words[label[-1]] + "</mark>"
    return " ".join(words)
            
        

def write_html(file_path, text):
    with open(file_path, "w") as hf:
        hf.write(text)
        hf.close()
    pass

def get_js_code():
    with open(HTML_DIR+"js_code.js", "r") as hf:
        js_code = hf.readlines()
        hf.close()
    return "".join(js_code)

def get_mark_style():
    with open(HTML_DIR+"mark.html", "r") as hf:
        mark_style = hf.readlines()
        hf.close()
    return "".join(mark_style)

def get_main_template():
    with open(HTML_DIR+"template.html", "r") as hf:
        main_template_html = hf.readlines()
        hf.close()
    return "".join(main_template_html)

def get_element_template():
    with open(HTML_DIR+"element_template.html", "r") as hf:
        element_template_html = hf.readlines()
        hf.close()
    return "".join(element_template_html)

def main():
    with open("eval_data/tagged_untagged_data.json", "r") as jf:
        orders = json.loads(jf.read())

    main_template_html = get_main_template()
    mark_style = get_mark_style()
    js_code = get_js_code()
    element_template_html = get_element_template()

    content_list = []
    a_page = []

    for order_index in range(len(orders)):
        label_list = orders[order_index]["label"]
        order_list = orders[order_index]["order"]

        for index in range(len(label_list)):
            label = label_list[index]
            sentence = order_list[index][1]
            name = order_list[index][0]

            if label:
                marked_text = mark_words(sentence, label) 
                marked_text = name + ":" + marked_text
                
                if index!=0 and index!=len(label_list)-1:
                    head_text = order_list[index-1][0]+":"+"\t"+order_list[index-1][1]
                    tail_text = order_list[index+1][0]+":"+"\t"+order_list[index+1][1]
                    marked_text = head_text + "<br>" + marked_text + "<br>" + tail_text
                elif index==0 and index!=len(label_list)-1:
                    tail_text = order_list[index+1][0]+":"+"\t"+order_list[index+1][1]
                    marked_text = marked_text + "<br>" + tail_text
                elif index!=0 and index==len(label_list)-1:
                    head_text = order_list[index-1][0]+":"+"\t"+order_list[index-1][1]
                    marked_text = head_text + "<br>" + marked_text
                
                # order中的第几句话
                a_element = element_template_html.format(len(a_page), marked_text, index, order_index)
                a_page.append(a_element)
            if len(a_page)==100:
                content_list.append(a_page)
                a_page = []
    print(len(content_list))

    for content_index in range(len(content_list)):
        if content_index < len(content_list)-1:
            next_page_name = "tagged_page"+str(content_index+1)+".html"
        else:
            next_page_name = "thanks_page.html"
        content_to_write = "".join(content_list[content_index])
        html_page = main_template_html.format(mark_style, 
                                              js_code,
                                              content_to_write,
                                              "label"+str(content_index)+".txt", next_page_name)
        with open(HTML_DIR+"tagged_page"+str(content_index)+".html", "w") as fout:
            fout.write(html_page)
            fout.close()

if __name__=="__main__":
    main()
