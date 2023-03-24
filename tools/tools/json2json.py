import json
import time

with open('D:/giga/gmot/logs/DINO/results_train.json') as infile:
    info = json.load(infile)


total_json = json.dumps(info, indent=2)
with open('D:/giga/gmot/logs/DINO/results_train1.json', "w", encoding='utf-8') as f:
    f.write(total_json)
    print("保存文件到本地完成")