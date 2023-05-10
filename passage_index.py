# %%
import json
import jieba
from collections import defaultdict
from tqdm import tqdm

stopwords = [w.strip() for w in open("./data/stopwords.txt").readlines()]

word2pid = defaultdict(dict)
with open("./data/passages_multi_sentences.json")as f:
    lines = f.readlines()
for line in tqdm(lines):
    data = json.loads(line)
    pid = data["pid"]
    for sent in data["document"]:
        words = jieba.lcut(sent)
        for w in words:
            if w not in stopwords:
                if pid in word2pid[w]:
                    word2pid[w][pid] += 1
                else:
                    word2pid[w][pid] = 1
# %%
import pickle
with open("data/word2pid.pkl", "wb")as f:
    f.write(pickle.dumps(word2pid))
# %%
