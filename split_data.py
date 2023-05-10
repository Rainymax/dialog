import random
random.seed(0)
with open("./data/train.json")as f:
    lines = f.readlines()
lines = [line.strip() for line in lines]
random.shuffle(lines)
train_size = int(len(lines) * 0.8)

with open("./data/train.json", "w")as fw:
    fw.writelines("\n".join(lines[:train_size]))

with open("./data/test.json", "w")as fw:
    fw.writelines("\n".join(lines[train_size:]))
