from email.policy import default
import pickle
from collections import Counter
import json
import jieba
import jieba.posseg
from collections import defaultdict

class SearchEngine:
    def __init__(self, file):
        with open(file, "rb")as f:
            self.word2pid = pickle.load(f)
    
    def search(self, keywords):
        pid2score = {}
        for w in keywords:
            pids = self.word2pid.get(w, [])
            # print(pids)
            for i in pids:
                if i in pid2score:
                    pid2score[i][0] += 1
                    pid2score[i][1] += len(self.word2pid) / len(pids) * pids[i]
                else:
                    pid2score[i] = [1, len(self.word2pid) / len(pids) * pids[i]]
        return list(sorted(pid2score.items(), key=lambda x:x[1], reverse=True))
        
        # pids = []
        # for w in keywords:
        #     pids += self.word2pid.get(w, [])
        # return list(sorted(dict(Counter(pids)).items(), key=lambda x:x[1], reverse=True))

def process_query(query, stopwords):
    words = jieba.lcut(query.strip())
    words = [w for w in words if w not in stopwords]
    words = list(set(words))
    return words

class DataLoader:
    def __init__(self, file="./data/train.json"):
        self.data = self.load_data(file)
    
    def load_data(self, file):
        data = []
        with open(file)as f:
            lines = f.readlines()
        for line in lines:
            data.append(json.loads(line))
        return data
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __iter__(self):
        return iter(self.data)
    
    def __len__(self):
        return len(self.data)
            


if __name__ == "__main__":
    engine = SearchEngine("./data/word2pid.pkl")
    dataloader = DataLoader()
    stopwords = [w.strip() for w in open("./data/stopwords.txt").readlines()]
    top1_correct = 0
    top3_correct = 0
    top10_correct = 0
    for data in dataloader:
        words = process_query(data["question"], stopwords)
        ans = data["pid"]
        preds = engine.search(words)
        if len(preds) < 1:
            print(data["question"])
            continue
        if preds[0][0] == ans:
            top1_correct += 1
            top3_correct += 1
        elif ans in [p[0] for p in preds[:3]]:
            top3_correct += 1
        if ans in [p[0] for p in preds[:10]]:
            top10_correct += 1
    print("Top1 acc=%.4f"%(top1_correct/len(dataloader)))
    print("Top3 acc=%.4f"%(top3_correct/len(dataloader)))
    print("Top10 acc=%.4f"%(top10_correct/len(dataloader)))




            
