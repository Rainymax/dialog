# 问答系统设计与实现

## Data Preparation
- 将下载下来的原数据放到`data`文件夹中

## How to Run
### 1. 文档检索
- 对所有文章分词并去除停用词后，建立word2pid的索引，每个词对应包含这个词的pid以及相应的该词出现的次数(passage_index.py)
- 对问题分词之后去除停用词并去重，得到若干关键词
- 搜索时，对每个问题中的关键词
   - 累计当前文档命中数量（命中一个关键词加一）
   - 累计包含该关键词的文档的分数，分数具体计算方法为IDF*该词在当前文档出现的频率
- 按照相似度返回排序后的文档
   - 相似度首先考虑命中数量，再考虑分数累计

```
python passage_index.py
python step1_search_passage.py
```

### 2. 构建问题分类器
- 使用`DMetaSoul/sbert-chinese-general-v1-distill`预训练模型加上MLP classification head训练

```
python step2_question_classification.py
```


### 3. 候选答案句
- 将数据集随即划分为9:1的两部分作为训练集和测试集
- 每篇文档中的正确答案句作为正样本，同一文档中的其他句子作为负样本
- 使用预训练模型`DMetaSoul/sbert-chinese-general-v1-distill`提取句子特征以及上一步训练好的encoder提取问题的特征
- 最终问题的embedding为二者逐点相乘的结果
- 再将问题embedding分别与正负样本候选句子的句子特征concatenate之后得到最终表示
- 使用InfoNCELoss优化模型

```
python split_data.py
python step3_sentence_sim.py
```

### 4. MRC抽取
- 训练集测试集划分跟前一步一致
- 采用边界模型的训练方式，使用`bert-base-chinese`预训练模型对每个token打两个分数，一个分数表示是否是答案起始位置，另一个表示是否为答案终止位置
- 使用交叉熵损失函数

```
python step4_mrc.py
```