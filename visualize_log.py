import json
import csv
import os

LOG_PATH = "./log/"
ALL_KEYS = ["Win", "TP", "FP", "FN", 
            "Precision", "Recall", "F1", "PredictTime", 
            "TP_A", "FP_A", "FN_A",
            "TP_B", "FP_B", "FN_B",
            "TP_C", "FP_C", "FN_C",
            "Precision_A", "Recall_A", "F1_A",
            "Precision_B", "Recall_B", "F1_B",
            "Precision_C", "Recall_C", "F1_C"
            ]

def modify_the_data(data_path):
    data = json.load(open(LOG_PATH+data_path, "r"))
    data_list = [["model_name:", data_path[:-4]]]
    for key in ALL_KEYS:
        data[key].reverse()
        data[key].append(key)
        data[key].reverse()
        data_list.append(data[key])
    return data_list

def main():
    with open('log_valid.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        dir_list = os.listdir(LOG_PATH)
        dir_list.sort()
        for file_path in dir_list:
            filter_list = []

            continue_flag = True
            for check_path in file_path:

                if check_path in file_path:
                    continue_flag = False
            if continue_flag:
                continue

            data_list = modify_the_data(file_path)
            for data_line in data_list:
                writer.writerow(data_line)
            writer.writerow([""])

if __name__=="__main__":
    main()
