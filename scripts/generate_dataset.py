# 形成train、val和test的数据集
# 使用karpathsplit给出的数据集以及预处理后包含object categories的ann_dict，最终整理为一个list
# 其中每个item: {"cocoid":xxx, caption":xxx, "categories":xxx, "filename":xxx, "split":xxx}
# 注意，由于ann_dict中cocoid不全，因此train、val和test都会少个别样例
import json
from tqdm import tqdm

dataset_coco_karpath = json.load(open('../data/dataset_coco.json', 'r'))["images"]
ann_dict = json.load(open('../data/ann_dict.json'))

dataset_train = []
dataset_val = []
dataset_test = []

num_cocoid_notfound = 0
for item in tqdm(dataset_coco_karpath):
    if item["split"] == "train" or item["split"] == "restval":
        if str(item["cocoid"]) not in ann_dict:
            num_cocoid_notfound += 1
            continue
        for sentence in item["sentences"]:
            item_new = {"cocoid": item["cocoid"], "caption": sentence["tokens"], "categories": ann_dict[str(item["cocoid"])],
                        "filename": item["filename"], "split": "train"}
            dataset_train.append(item_new)
    else:
        if str(item["cocoid"]) not in ann_dict:
            num_cocoid_notfound += 1
            continue
        captions = [sentence["tokens"] for sentence in item["sentences"]]
        item_new = {"cocoid": item["cocoid"], "caption": captions, "categories": ann_dict[str(item["cocoid"])],
                    "filename": item["filename"], "split": item["split"]}
        if item["split"] == "val":
            dataset_val.append(item_new)
        elif item["split"] == "test":
            dataset_test.append(item_new)

print("Num of train: " + str(len(dataset_train)))
print("Num of val: " + str(len(dataset_val)))
print("Num of test: " + str(len(dataset_test)))
print("cocoid not found: " + str(num_cocoid_notfound))
with open('../data/dataset_train.json', 'w') as f:
    json.dump(dataset_train, f)
with open('../data/dataset_val.json', 'w') as f:
    json.dump(dataset_val, f)
with open('../data/dataset_test.json', 'w') as f:
    json.dump(dataset_test, f)
