import json
import re

def write_bytes_name(fin):
    name_list = []
    for line in fin:
        name_get = re.findall(".*\t(.*)\t:.*", line)
        if name_get:
            name = name_get[0]
            
            if name not in name_list:
                name_list.append(name)
    
    with open("eval_data/advisor_name_byte.json","w+") as fout:
        json_str = json.dumps(name_list, indent=2)
        fout.write(json_str)

def main():
    with open("eval_data/order.txt", "r") as fin:
        write_bytes_name(fin)

if __name__=="__main__":
    main()
