from cmath import e
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel
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
from torch.nn.parallel import DistributedDataParallel as DDP



class BertForQuestionAnswering(nn.Module):


    def __init__(self):
        nn.Module.__init__(self)
        self.bert = BertModel.from_pretrained("bert-base-chinese")
        self.qa_outputs = nn.Linear(768, 2)


    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        token_type_ids = None,
        position_ids = None,
        head_mask = None,
        inputs_embeds = None,
        start_positions = None,
        end_positions = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
    ):
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """


        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        return total_loss, start_logits, end_logits

class Instance:
    def __init__(self, data):
        self.q = data["question"]
        self.text = "".join(data["answer_sentence"])
        self.answer = data["answer"].strip()
        self.get_pos()
    
    def get_pos(self):
        self.start_pos = self.text.find(self.answer)
        self.end_pos = self.start_pos + len(self.answer)
    
    def valid(self):
        return self.start_pos != -1
    
    def tokenize(self, tokenizer, max_length=128):
        query_tokenized = tokenizer.tokenize(self.q)
        text1_tokenized = tokenizer.tokenize(self.text[:self.start_pos])
        answer_tokenized = tokenizer.tokenize(self.answer)
        text2_tokenized = tokenizer.tokenize(self.text[self.end_pos:])
        self.input_ids = tokenizer.convert_tokens_to_ids([tokenizer.cls_token] + query_tokenized + [tokenizer.sep_token] + text1_tokenized + answer_tokenized + text2_tokenized + [tokenizer.sep_token])
        self.type_token_ids = [0] * (len(query_tokenized) + 2) + [1] * (len(self.input_ids) - len(query_tokenized) - 2)
        self.attention_mask = [1] * len(self.input_ids)

        self.start_pos = len(query_tokenized) + 2 + len(text1_tokenized)
        self.end_pos = self.start_pos + len(answer_tokenized)

        self.input_ids = self.input_ids[:max_length]
        self.type_token_ids = self.type_token_ids[:max_length]
        self.attention_mask = self.attention_mask[:max_length]

        while len(self.input_ids) < max_length:
            self.input_ids.append(tokenizer.pad_token_id)
            self.attention_mask.append(0)
            self.type_token_ids.append(0)
        assert len(self.input_ids) == len(self.attention_mask) == len(self.type_token_ids)

        self.input_ids = torch.LongTensor(self.input_ids)
        self.attention_mask = torch.LongTensor(self.attention_mask)
        self.type_token_ids = torch.LongTensor(self.type_token_ids)

        # self.start_pos = torch.LongTensor([self.start_pos])
        # self.end_pos = torch.LongTensor([self.end_pos])
    
    def data(self):
        return {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "token_type_ids": self.type_token_ids,
            "start_positions": self.start_pos,
            "end_positions": self.end_pos
        }, self.answer

class myDataset(Dataset):
    def __init__(self, lines, tokenizer, max_length=128):
        self.samples = self.load_samples(lines)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenize()
    
    def load_samples(self, json_lines):
        samples = []
        for line in tqdm(json_lines, desc="loading"):
            if line.strip():
                sample = Instance(json.loads(line))
                if sample.valid():
                    samples.append(sample)
        return samples
    
    def tokenize(self):
        for sample in tqdm(self.samples, desc="tokenize"):
            sample.tokenize(self.tokenizer, self.max_length)
    
    def __getitem__(self, index):
        return self.samples[index].data()
    
    def __len__(self):
        return len(self.samples)

def f1(p, r):
    if p ==0 or r == 0:
        return 0
    return 2 * (p * r) / (p + r)

def get_metrics(true_pos, pred_pos):
    assert len(true_pos) == len(pred_pos)
    em = 0
    correct = 0
    pred = 0
    true = 0
    for i in range(len(true_pos)):
        t = true_pos[i]
        p = pred_pos[i]
        if t[0] == p[0] and t[1] == p[1]:
            em += 1 
            correct += t[1] - t[0]
        else:
            if p[1] - p[0] <= 0:
                pass
            else:
                correct += len(set(range(t[0], t[1])).intersection(set(range(p[0], p[1]))))
        pred += np.max(p[1] - p[0], 0)
        true += t[1] - t[0]
    precision = correct / pred
    recall = correct / true
    return {"EM": em / len(true_pos), "precision": precision, "recall": recall, "f1": f1(precision, recall)}

def evaluate(model, dataloader):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for inputs, answers in dataloader:
            for k in inputs:
                inputs[k] = inputs[k].to(device)
            # print(inputs)
            # print(answers)
            loss, start_logits, end_logits = model(**inputs)

            start_pred = torch.argmax(start_logits, dim=-1).cpu().numpy().tolist()
            end_pred = torch.argmax(end_logits, dim=-1).cpu().numpy().tolist()

            pred_pos = list(zip(start_pred, end_pred))
            true_pos = list(zip(inputs["start_positions"].cpu().numpy().tolist(), inputs["end_positions"].cpu().numpy().tolist()))

            preds += pred_pos
            trues += true_pos
    return get_metrics(trues, preds)

if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--local_rank", type=int, default=-1)
    # args = parser.parse_args()
    # import os
    # local_rank = int(os.environ["LOCAL_RANK"])

    # torch.cuda.set_device(local_rank)
    device = torch.device('cuda')
    # torch.distributed.init_process_group(backend='nccl')

    cuda = True
    
    epochs = 5
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    with open("./data/train.json")as f:
        train_lines = f.readlines()
    with open("./data/test.json")as f:
        test_lines = f.readlines()

    train_dataset = myDataset(train_lines, tokenizer)
    test_dataset = myDataset(test_lines, tokenizer)
    # print(test_dataset[0])
    # train_sampler = DistributedSampler(train_dataset)

    train_dataloader = DataLoader(train_dataset, batch_size=8)

    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    model = BertForQuestionAnswering()
    model.to(device)
    # model = DDP(model, find_unused_parameters=True)

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=len(train_dataloader) * epochs)

    best_metric = 0.0
    for epoch in range(epochs):
        preds = []
        trues = []
        losses = []
        for inputs, answers in tqdm(train_dataloader, desc=f"Epoch {epoch}"):
            for k in inputs:
                inputs[k] = inputs[k].to(device)

            model.train()
            # print(inputs)
            # print(answers)
            loss, start_logits, end_logits = model(**inputs)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # if local_rank == 0:
            losses.append(loss.item())

            start_pred = torch.argmax(start_logits, dim=-1).cpu().numpy().tolist()
            end_pred = torch.argmax(end_logits, dim=-1).cpu().numpy().tolist()

            pred_pos = list(zip(start_pred, end_pred))
            true_pos = list(zip(inputs["start_positions"].cpu().numpy().tolist(), inputs["end_positions"].cpu().numpy().tolist()))

            preds += pred_pos
            trues += true_pos
        # if local_rank == 0:
        train_metric = get_metrics(trues, preds)
        test_metric = evaluate(model, test_dataloader)
        print("Epoch %d, train loss=%.4f, train metric=%s, test metric=%s" % (epoch, np.mean(losses), json.dumps(train_metric), json.dumps(test_metric)))

        if test_metric["f1"] > best_metric:
            best_metric = test_metric["f1"]
            print("best!")
            torch.save(model.state_dict(), "model/step4.pt")

        
