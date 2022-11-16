import json
import csv
import os

LOG_PATH = "./log/"
ALL_KEYS = ["Win", "TP", "FP", "FN", "Precision", "Recall", "F1", "PredictTime"]

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
    with open('log.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        dir_list = os.listdir(LOG_PATH)
        dir_list.sort()
        for file_path in dir_list:
            data_list = modify_the_data(file_path)
            for data_line in data_list:
                writer.writerow(data_line)
            writer.writerow([""])

if __name__=="__main__":
    main()
