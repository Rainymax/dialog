from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
import json
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
from collections import defaultdict
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler

def get_query_idx(splits):
    query_idx = []
    query_idx.append(0)
    current = 0
    for l in splits[:-1]:
        query_idx.append(current+l)
        current += l
    return query_idx

class myModel(nn.Module):
    def __init__(self, feature_encoder, encoder):
        nn.Module.__init__(self)
        self.feature_encoder = feature_encoder
        self.encoder = encoder
        self.linear = nn.Linear(768*2, 768)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(768, 1)
        self.softmax = nn.Softmax(0)
        self.dropout = nn.Dropout(0.5)

    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def forward(self, input_ids, attention_mask, query_input_ids, query_attention_mask):


        # question feature
        output = self.feature_encoder(query_input_ids, query_attention_mask)
        feature = self.mean_pooling(output, query_attention_mask)

        # sentence feature
        sentence_embeddings = self.mean_pooling(self.encoder(input_ids, attention_mask), attention_mask)

        feature = self.dropout(feature)
        sentence_embeddings = self.dropout(sentence_embeddings)
        return feature, sentence_embeddings

def get_scores(feature, sentence_embeddings, query_idx):
    # q_embed, q_feat, c_embed
    out = []
    query_idx.append(len(sentence_embeddings))
    for j, idx in enumerate(query_idx[:-1]):
        q_embed = sentence_embeddings[idx:idx+1]
        q_feat = feature[j:j+1]
        c_embed = sentence_embeddings[idx+1:query_idx[j+1]]

        # c_embed = self.dropout(c_embed)
        q_feat_new = q_embed.mul(q_feat)
        # q_feat_new = self.dropout(q_feat_new)
        out.append(F.softmax(cosine_sim(q_feat_new, c_embed), dim=0))

    out = torch.cat(out, dim=0).squeeze(-1)
    return out

class Instance:
    def __init__(self, data, pid2sentences):
        self.q = data["question"]
        self.candidates = pid2sentences[data["pid"]]
        ans = list(set([s.replace(" ", "") for s in data["answer_sentence"]]))
        self.pos_idx = [i for i in range(len(self.candidates)) if self.candidates[i].replace(" ", "") in ans]
        # if not self.pos_idx:
        # if len(self.pos_idx) != len(ans):
        #     print(self.pos_idx)
        #     print(data)
    
    def tokenize(self, tokenizer, max_length):
        self.tokenized = tokenizer([self.q] + self.candidates, max_length=max_length, return_tensors="pt", padding="max_length", truncation=True)
        self.labels = []
        for i in range(len(self.candidates)):
            if i in self.pos_idx:
                self.labels.append(1)
            else:
                self.labels.append(0)
        assert len(self.labels) == len(self.candidates) 

    def data(self):
        return {
            **self.tokenized,
            "pos_idx": self.pos_idx,
            "sent_num": len(self.candidates) + 1,
            "labels": self.labels
        }

class myDataset(Dataset):
    def __init__(self, json_lines, pid2sentences, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_samples(json_lines, pid2sentences)
        self.tokenize()
    
    def load_samples(self, json_lines, pid2sentences):
        samples = []
        for line in tqdm(json_lines, desc="loading"):
            if line.strip():
                sample = Instance(json.loads(line), pid2sentences)
                samples.append(sample)
        return samples
    
    def tokenize(self):
        for sample in tqdm(self.samples, desc="tokenize"):
            sample.tokenize(self.tokenizer, max_length=self.max_length)
    
    def __getitem__(self, index):
        return self.samples[index].data()

    def __len__(self):
        return len(self.samples)

def collator(data_list):
    collate_data = defaultdict(list)
    for data in data_list:
        for k in data:
            if k == "labels":
                collate_data[k] += data[k]
            else:
                collate_data[k].append(data[k])
    for k in collate_data:
        if isinstance(collate_data[k][0], torch.Tensor):
            try:
                collate_data[k] = torch.cat(collate_data[k], dim=0)
            except:
                print(k)
                print(collate_data[k][0].size(), collate_data[k][1].size())
    collate_data["labels"] = torch.LongTensor(collate_data["labels"])
    return collate_data

def load_pid2sentences():
    pid2sentences= {}
    with open("./data/passages_multi_sentences.json")as f:
        lines = f.readlines()
    for line in lines:
        if line.strip():
            data = json.loads(line)
            sents = []
            for s in data["document"]:
                s = s.replace(" ", "")
                if s in sents:
                    continue
                sents.append(s)
            pid2sentences[data["pid"]] = sents
    return pid2sentences

def cosine_sim(embed1, embed2):
    embed1 = nn.functional.normalize(embed1, dim=1)
    embed2 = nn.functional.normalize(embed2, dim=1)
    sent_sim = torch.matmul(embed1.unsqueeze(1), embed2.unsqueeze(-1)).squeeze(1) # (cand_num, anchor_num)
    return sent_sim

# def cosine_sim(sent_embed):
#     sent_embed = nn.functional.normalize(sent_embed, dim=1)
#     anchor_embed = sent_embed[0:1]
#     cand_embed = sent_embed[1:]
#     sent_sim = torch.matmul(anchor_embed.unsqueeze(1), cand_embed.unsqueeze(-1)).squeeze(1) # (cand_num, anchor_num)
#     # print(sent_sim.size())
#     return sent_sim

class ContrastiveLoss(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.margin = torch.tensor(0.01)

    def contrastive_loss(self, pos_sim, neg_sim):
        loss = torch.tensor(0.0)
        global cuda
        if cuda:
            loss = loss.cuda()
        # anchor_embed = nn.functional.normalize(anchor_embed, dim=1)
        # pos_embed = nn.functional.normalize(pos_embed, dim=1)
        # neg_embed = nn.functional.normalize(neg_embed, dim=1)
        # pos_sim = torch.matmul(anchor_embed.unsqueeze(1), pos_embed.unsqueeze(-1))
        # neg_sim = torch.matmul(anchor_embed.unsqueeze(1), neg_embed.unsqueeze(-1))
        for score1 in pos_sim:
            for score2 in neg_sim:
                loss += torch.clamp(score2 - score1 + self.margin, min=0.0).sum()
        return loss



    def forward(self, sent_embeddings, splits, pos_idx):
        loss = torch.tensor(0.0)
        global cuda
        if cuda:
            loss = loss.cuda()
        total = 0
        current_l = 0
        for j, l in enumerate(splits):
            embed = sent_embeddings[current_l:current_l+l]
            # anchor_embed = embed[0:1]
            # cand_embed = embed[1:]
            # pos_embed = cand_embed[pos_idx[j]]
            # neg_embed = cand_embed[[i for i in range(len(cand_embed)) if i not in pos_idx]]
            cand_sim = cosine_sim(embed)
            pos_sim = cand_sim[pos_idx[j]]
            neg_sim = cand_sim[[i for i in range(len(cand_sim)) if i not in pos_idx[j]]]

            loss += self.contrastive_loss(pos_sim, neg_sim)
            total += len(pos_sim) * len(neg_sim)
        return loss / total

def get_mrr(logits, splits, pos_idx):
    mrr = 0.0
    current_l = 0
    for j, l in enumerate(splits):
        scores = logits[current_l:current_l+l-1]
        current_l += l - 1
        ranks = torch.argsort(scores, descending=True)
        for rank, sample_index in enumerate(ranks):
            if sample_index in pos_idx[j]:
                mrr += 1.0 / (rank + 1.0)
                # break
    return mrr

def evaluate(model, dataloader):
    model.eval()
    test_metric = 0.0
    total = 0
    with torch.no_grad():
        for data in tqdm(dataloader):
            if cuda:
                for k in data:
                    if isinstance(data[k], torch.Tensor):
                        data[k] = data[k].cuda()
            total += len(data["sent_num"])
            query_idx = get_query_idx(data["sent_num"])
            feature, sentence_embeddings = model(data["input_ids"], data["attention_mask"], data["input_ids"][query_idx], data["attention_mask"][query_idx])
            logits = get_scores(feature, sentence_embeddings, query_idx)
            # logits = F.softmax(output, dim=-1)
            # sentence_embeddings = mean_pooling(output, data['attention_mask'])
            test_metric += get_mrr(logits, data["sent_num"], data["pos_idx"])
    return test_metric / total

if __name__ == "__main__":
    import random
    from tqdm import tqdm
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--local_rank", type=int, default=-1)
    # args = parser.parse_args()

    # torch.cuda.set_device(args.local_rank)
    # device = torch.device('cuda', 0)
    # torch.distributed.init_process_group(backend='nccl')

    cuda = True
    random.seed(0)
    with open("./data/train.json")as f:
        train_lines = f.readlines()
    
    with open("./data/test.json")as f:
        test_lines = f.readlines()

    tokenizer = AutoTokenizer.from_pretrained('DMetaSoul/sbert-chinese-general-v1-distill')
    encoder = AutoModel.from_pretrained('DMetaSoul/sbert-chinese-general-v1-distill')
    # tokenizer = AutoTokenizer.from_pretrained('DMetaSoul/sbert-chinese-general-v2')
    # encoder = AutoModel.from_pretrained('DMetaSoul/sbert-chinese-general-v2')
    feature_encoder = AutoModel.from_pretrained('DMetaSoul/sbert-chinese-general-v1-distill')
    feature_encoder.load_state_dict(torch.load("./model/step2_encoder.pt"))
    model = myModel(feature_encoder, encoder)
    for p in model.feature_encoder.parameters():
        p.requires_grad = False

    

    pid2sents = load_pid2sentences()
    train_dataset = myDataset(train_lines, pid2sents, tokenizer)
    # train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=4, collate_fn = collator)

    test_dataset = myDataset(test_lines, pid2sents, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=4, collate_fn = collator)

    # Loss = ContrastiveLoss()
    # Loss = nn.CrossEntropyLoss()
    # if cuda:
    #      Loss = Loss.cuda()

    epochs = 20
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=len(train_dataloader) * epochs)

    if cuda:
        model = model.cuda()
        # model.to(device)
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    best_score = 0.0
    for epoch in range(epochs):
        labels = []
        preds = []
        epoch_loss = []
        train_metric = 0.0
        for data in tqdm(train_dataloader):
            if cuda:
                for k in data:
                    if isinstance(data[k], torch.Tensor):
                        data[k] = data[k].cuda()

            query_idx = get_query_idx(data["sent_num"])
            feature, sentence_embeddings = model(data["input_ids"], data["attention_mask"], data["input_ids"][query_idx], data["attention_mask"][query_idx])
            logits = get_scores(feature, sentence_embeddings, query_idx)
            # logits = model(data, data["sent_num"])
            # logits = F.softmax(out, dim=-1)
            loss =  - torch.log(logits[data["labels"]==1]).mean()
            # loss = Loss(logits, data["labels"])
            # labels += data["labels"].cpu().numpy().tolist()
            # preds += torch.argmax(logits, dim=-1).cpu().numpy().tolist()
            # sentence_embeddings = mean_pooling(output, data['attention_mask'])
            train_metric += get_mrr(logits, data["sent_num"], data["pos_idx"])

            # loss = Loss(sentence_embeddings, data["sent_num"], data["pos_idx"])

            # del sentence_embeddings

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            epoch_loss.append(loss.item())
        # acc = accuracy_score(labels, preds)
        # print("Epoch %d, train loss=%.4f, train MRR=%.4f" % (epoch, np.mean(epoch_loss), train_metric / len(train_dataset)))
        # if args.local_rank == 0:
        test_metric = evaluate(model, test_dataloader)

        print("Epoch %d, train loss=%.4f, train MRR=%.4f, test MRR=%.4f" % (epoch, np.mean(epoch_loss), train_metric / len(train_dataset), test_metric))
        if test_metric > best_score:
            print("best!")
            best_score = test_metric
            torch.save(model.state_dict(), "./model/step3.pt")
            # torch.save(model.encoder.state_dict(), "./model/step3_encoder.pt")


            # del loss
            # print("6:", torch.cuda.memory_allocated())

            # torch.cuda.empty_cache()


# %%
