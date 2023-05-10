from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import torch
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn


# #Mean Pooling - Take attention mask into account for correct averaging
# def mean_pooling(model_output, attention_mask):
#     token_embeddings = model_output[0] #First element of model_output contains all token embeddings
#     input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#     return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class myModel(nn.Module):
    def __init__(self, encoder, label_num=10):
        nn.Module.__init__(self)
        self.encoder = encoder
        self.cls = nn.Linear(768, label_num)
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def forward(self, inputs):
        output = self.encoder(**inputs)
        sentence_embed = self.mean_pooling(output, inputs["attention_mask"])
        logits = self.cls(sentence_embed)
        return logits


class myDataset(Dataset):
    def __init__(self, file, label2id=None):
        # self.tokenizer = tokenizer
        # self.max_length = max_length
        self.label2id = label2id
        self.labels, self.data = self.load_data(file)
        self.convert_labels()
        # self.tokenize()

    def load_data(self, file):
        df = pd.read_csv(file, sep="\t", header=None)
        labels = df[0]
        data = df[1]
        return list(labels), list(data)
    
    def convert_labels(self):
        if self.label2id is None:
            label_set = list(sorted(set(self.labels)))
            self.label2id = {l:i for i, l in enumerate(label_set)}
            self.id2label = {i:l for i, l in enumerate(label_set)}
        self.labels = [self.label2id[l] for l in self.labels]
    
    def tokenize(self):
        self.tokenizer(self.data, max_length=self.max_length, return_tensors="pt")

    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    
    def __len__(self):
        return len(self.data)
    
def collator(data):
    return zip(*data)

def evaluate(model, dataloader):
    epoch_label = []
    epoch_pred = []
    model.eval()
    with torch.no_grad():
        for data, label in tqdm(dataloader):
            # print(data)
            # print(label)
            label = torch.LongTensor(label).to(device)
            inputs = tokenizer(list(data), max_length=64, padding=True, return_tensors="pt")
            for k in inputs:
                inputs[k] = inputs[k].to(device)
            logits = model(inputs)
            epoch_label += label.cpu().numpy().tolist()
            epoch_pred += torch.argmax(logits, dim=-1).cpu().numpy().tolist()
    return accuracy_score(epoch_label, epoch_pred)

def test(model, dataloader, id2label):
    epoch_label = []
    epoch_pred = []
    model.eval()
    with torch.no_grad():
        for data, label in tqdm(dataloader):
            # print(data)
            # print(label)
            label = torch.LongTensor(label).to(device)
            inputs = tokenizer(list(data), max_length=64, padding=True, return_tensors="pt")
            for k in inputs:
                inputs[k] = inputs[k].to(device)
            logits = model(inputs)
            epoch_label += label.cpu().numpy().tolist()
            epoch_pred += torch.argmax(logits, dim=-1).cpu().numpy().tolist()
    coarse_label = [id2label[label].split("_")[0] for label in epoch_label]
    coarse_pred = [id2label[pred].split("_")[0] for pred in epoch_pred]
    coarse_acc = np.sum([coarse_label[i] == coarse_pred[i] for i in range(len(coarse_label))]) / len(coarse_label)
    return accuracy_score(epoch_label, epoch_pred), coarse_acc


if __name__ == "__main__":
    device="cuda"
    epochs = 10
    lr = 2e-5
    do_train = True

    train_dataset = myDataset("./data/train_questions.txt")
    train_dataloader = DataLoader(train_dataset, collate_fn=collator, batch_size=16, shuffle=True)
    test_dataset = myDataset("./data/test_questions.txt")
    test_dataloader = DataLoader(test_dataset, collate_fn=collator, batch_size=100, shuffle=False)
    id2label = train_dataset.id2label

    tokenizer = AutoTokenizer.from_pretrained('DMetaSoul/sbert-chinese-general-v1-distill')
    encoder = AutoModel.from_pretrained('DMetaSoul/sbert-chinese-general-v1-distill')

    model = myModel(encoder, label_num=len(id2label))

    model = model.to(device)
    total_steps = epochs * len(train_dataloader)

    Loss = torch.nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.2 * total_steps, num_training_steps=total_steps)

    if do_train:
        best_acc = 0.0
        for e in range(epochs):
            model.train()
            epoch_loss = []
            epoch_label = []
            epoch_pred = []
            for data, label in tqdm(train_dataloader):
                # print(data)
                # print(label)
                label = torch.LongTensor(label).to(device)
                inputs = tokenizer(list(data), max_length=64, padding=True, return_tensors="pt")
                for k in inputs:
                    inputs[k] = inputs[k].to(device)
                logits = model(inputs)                
                loss = Loss(logits, label)
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                epoch_loss.append(loss.item())
                epoch_label += label.cpu().numpy().tolist()
                epoch_pred += torch.argmax(logits, dim=-1).cpu().numpy().tolist()
                # print(loss.item())
            print("Epoch %d, loss=%.4f, acc=%.4f" % (e, np.mean(epoch_loss), accuracy_score(epoch_label, epoch_pred)))
            acc = evaluate(model, test_dataloader)
            print("Eval Epoch %d, acc=%.4f" % (e, acc))
            if acc > best_acc:
                best_acc = acc
                print("best!")
                import os
                if not os.path.exists("model"):
                    os.mkdir("model")
                torch.save(model.state_dict(), "./model/step2.pt")
                torch.save(model.encoder.state_dict(), "./model/step2_encoder.pt")

    
    model.load_state_dict(torch.load("./model/step2.pt"))
    acc, coarse_acc = test(model, test_dataloader, id2label)
    print("Test acc=%.4f, coarse_acc=%.4f" % (acc, coarse_acc))
