# -*- coding: utf-8 -*-
# 作业三：多模态特征融合预测（加深深度学习与注意力机制融合）

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import BertTokenizer, BertModel, AdamW
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import joblib

# ✅ 文本预处理函数（与LDA一致）
CUSTOM_FILTERS = [lambda x: x.lower(), strip_punctuation, strip_numeric, remove_stopwords, strip_short]
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    return text

def preprocess(texts):
    return [preprocess_string(clean_text(doc), CUSTOM_FILTERS) for doc in texts]

# ✅ 加载数据（替换为你的文件名）
df = pd.read_excel("task1_results.xlsx")

# ✅ 加载LDA模型和字典（路径替换为你的保存路径）
lda_model = joblib.load("lda_model.joblib")
dictionary = joblib.load("lda_dictionary.joblib")

# ✅ 预处理文本、获取主题向量
print("提取主题分布向量...")
texts = preprocess(df["text"].tolist())
corpus = [dictionary.doc2bow(text) for text in texts]

# 将主题分布向量转为定长
def get_topic_vector(bow, lda_model, num_topics):
    topic_dist = lda_model.get_document_topics(bow, minimum_probability=0)
    return np.array([prob for _, prob in sorted(topic_dist)])

topic_vectors = np.array([get_topic_vector(bow, lda_model, lda_model.num_topics) for bow in corpus])

# ✅ 情感 one-hot 编码
def encode_sentiment(sent):
    if sent == "正面": return [1,0,0]
    elif sent == "中性": return [0,1,0]
    elif sent == "负面": return [0,0,1]
    else: return [0,0,0]

sentiment_vectors = np.array([encode_sentiment(s) for s in df["情感倾向"]])

# ✅ 拼接多模态特征
total_features = np.hstack([topic_vectors, sentiment_vectors])
labels = df["label"].apply(lambda x: 0 if x == "fake" else 1).values

# ✅ 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(total_features, labels, test_size=0.2, random_state=42)

# ✅ 自定义数据集类（BERT + 主题 + 情感）
class MultiModalDataset(Dataset):
    def __init__(self, texts, topic_vectors, sentiment_vectors, labels, tokenizer, max_length=128):
        self.texts = texts
        self.topic_vectors = topic_vectors
        self.sentiment_vectors = sentiment_vectors
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        topic = self.topic_vectors[idx]
        sentiment = self.sentiment_vectors[idx]
        label = self.labels[idx]

        # 处理文本
        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        # 合并所有特征
        return input_ids, attention_mask, torch.tensor(topic, dtype=torch.float32), torch.tensor(sentiment, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# ✅ 使用 BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# ✅ 模型：融合 BERT、主题向量、情感向量的模型
class MultiModalBERT(nn.Module):
    def __init__(self, num_topics, num_sentiments):
        super(MultiModalBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.topic_fc = nn.Linear(num_topics, 128)
        self.sentiment_fc = nn.Linear(num_sentiments, 128)
        self.fc = nn.Linear(self.bert.config.hidden_size + 128 + 128, 2)

    def forward(self, input_ids, attention_mask, topic_vectors, sentiment_vectors):
        # BERT
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = bert_output.pooler_output  # 获取[CLS] token的输出

        # 主题和情感的线性变换
        topic_output = torch.relu(self.topic_fc(topic_vectors))
        sentiment_output = torch.relu(self.sentiment_fc(sentiment_vectors))

        # 拼接所有特征
        combined_output = torch.cat((cls_output, topic_output, sentiment_output), dim=1)

        # 分类层
        return self.fc(combined_output)

# ✅ 数据加载器
train_dataset = MultiModalDataset(X_train, topic_vectors[:len(X_train)], sentiment_vectors[:len(X_train)], y_train, tokenizer)
test_dataset = MultiModalDataset(X_test, topic_vectors[len(X_train):], sentiment_vectors[len(X_train):], y_test, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

# ✅ 初始化模型和优化器
model = MultiModalBERT(num_topics=lda_model.num_topics, num_sentiments=3).to('cuda' if torch.cuda.is_available() else 'cpu')
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()

# ✅ 训练函数
def train_epoch(model, data_loader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for input_ids, attention_mask, topic_vectors, sentiment_vectors, labels in tqdm(data_loader, desc='Training'):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        topic_vectors = topic_vectors.to(device)
        sentiment_vectors = sentiment_vectors.to(device)
        labels = labels.to(device)

        # 前向传播
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, topic_vectors, sentiment_vectors)
        loss = loss_fn(outputs, labels)

        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(data_loader), correct / total

# ✅ 评估函数
def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for input_ids, attention_mask, topic_vectors, sentiment_vectors, labels in tqdm(data_loader, desc='Evaluating'):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            topic_vectors = topic_vectors.to(device)
            sentiment_vectors = sentiment_vectors.to(device)
            labels = labels.to(device)

            outputs = model(input_ids, attention_mask, topic_vectors, sentiment_vectors)
            _, preds = torch.max(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total

# ✅ 训练和评估
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 3

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, loss_fn)
    print(f"Training loss: {train_loss:.4f}, accuracy: {train_acc:.4f}")

    val_acc = evaluate(model, test_loader)
    print(f"Validation accuracy: {val_acc:.4f}")

# ✅ 保存模型
joblib.dump(model, "multimodal_bert_model.joblib")
