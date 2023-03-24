import json

with open('D:/GMOT/gmot/logs/DINO/results_tr.json') as infile:
    info = json.load(infile)

pred_list = []
for item in info:
    if item['score'] > 0:
        item['category_id'] = 0
        item['bbox'][0] -= item['bbox'][2] / 2
        item['bbox'][1] -= item['bbox'][3] / 2
        pred_list.append(item)

total_json = json.dumps(pred_list, indent=2)
with open('D:/GMOT/gmot/logs/DINO/test_result.json', "w", encoding='utf-8') as f:
    f.write(total_json)
    print("保存文件到本地完成")