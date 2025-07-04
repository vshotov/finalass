# -*- coding: utf-8 -*-
# 多模态情感+主题+BERT融合预测（引入注意力机制）

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from gensim.parsing.preprocessing import preprocess_string, strip_punctuation, strip_numeric, remove_stopwords, strip_short
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import joblib
from tqdm import tqdm

# ===================== 文本预处理 =====================
CUSTOM_FILTERS = [lambda x: x.lower(), strip_punctuation, strip_numeric, remove_stopwords, strip_short]
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    return text

def preprocess(texts):
    return [preprocess_string(clean_text(doc), CUSTOM_FILTERS) for doc in texts]

# ===================== 加载数据 =====================
df = pd.read_excel("task1_results.xlsx")
lda_model = joblib.load("lda_model.joblib")
dictionary = joblib.load("lda_dictionary.joblib")

# 文本清洗
texts = preprocess(df["text"].tolist())
corpus = [dictionary.doc2bow(text) for text in texts]

# ===================== 构造主题向量 =====================
def get_topic_vector(bow, lda_model, num_topics):
    topic_dist = lda_model.get_document_topics(bow, minimum_probability=0)
    return np.array([prob for _, prob in sorted(topic_dist)])

topic_vectors = np.array([get_topic_vector(bow, lda_model, lda_model.num_topics) for bow in corpus])

# ===================== 情感向量 =====================
def encode_sentiment(sent):
    if sent == "正面": return [1,0,0]
    elif sent == "中性": return [0,1,0]
    elif sent == "负面": return [0,0,1]
    else: return [0,0,0]

sentiment_vectors = np.array([encode_sentiment(s) for s in df["情感倾向"]])

# ===================== 标签与特征划分 =====================
labels = df["label"].apply(lambda x: 0 if x == "fake" else 1).values
X_train_text, X_test_text, X_train_topic, X_test_topic, X_train_sent, X_test_sent, y_train, y_test = train_test_split(
    df['text'].tolist(), topic_vectors, sentiment_vectors, labels, test_size=0.2, random_state=42)

# ===================== 数据集类 =====================
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

        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        return input_ids, attention_mask, torch.tensor(topic, dtype=torch.float32), torch.tensor(sentiment, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# ===================== 注意力融合模块 =====================
class AttentionFusion(nn.Module):
    def __init__(self, query_dim, key_value_dim, hidden_dim):
        super(AttentionFusion, self).__init__()
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(key_value_dim, hidden_dim)
        self.value_proj = nn.Linear(key_value_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5

    def forward(self, query, key_value):
        q = self.query_proj(query).unsqueeze(1)
        k = self.key_proj(key_value)
        v = self.value_proj(key_value)
        attn_scores = torch.matmul(q, k.transpose(1, 2)) / self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_weights, v).squeeze(1)
        return context

# ===================== 融合模型 =====================
class MultiModalBERTWithAttention(nn.Module):
    def __init__(self, num_topics, num_sentiments, hidden_size=128):
        super(MultiModalBERTWithAttention, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.topic_fc = nn.Linear(num_topics, hidden_size)
        self.sentiment_fc = nn.Linear(num_sentiments, hidden_size)
        self.attn_fusion = AttentionFusion(query_dim=768, key_value_dim=hidden_size, hidden_dim=hidden_size)
        self.classifier = nn.Linear(768 + hidden_size, 2)

    def forward(self, input_ids, attention_mask, topic_vectors, sentiment_vectors):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = bert_output.pooler_output
        topic_emb = torch.relu(self.topic_fc(topic_vectors))
        sentiment_emb = torch.relu(self.sentiment_fc(sentiment_vectors))
        kv_stack = torch.stack([topic_emb, sentiment_emb], dim=1)
        fusion_vector = self.attn_fusion(cls_output, kv_stack)
        final_repr = torch.cat([cls_output, fusion_vector], dim=1)
        return self.classifier(final_repr)

# ===================== 模型训练 =====================
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_dataset = MultiModalDataset(X_train_text, X_train_topic, X_train_sent, y_train, tokenizer)
test_dataset = MultiModalDataset(X_test_text, X_test_topic, X_test_sent, y_test, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiModalBERTWithAttention(num_topics=lda_model.num_topics, num_sentiments=3).to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()

def train_epoch(model, data_loader):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for input_ids, attention_mask, topic_vectors, sentiment_vectors, labels in tqdm(data_loader, desc='Training'):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        topic_vectors = topic_vectors.to(device)
        sentiment_vectors = sentiment_vectors.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, topic_vectors, sentiment_vectors)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(data_loader), correct / total

def evaluate(model, data_loader):
    model.eval()
    correct, total = 0, 0
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

# ===================== 正式训练 =====================
epochs = 3
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    train_loss, train_acc = train_epoch(model, train_loader)
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    val_acc = evaluate(model, test_loader)
    print(f"Val Acc: {val_acc:.4f}")

# 保存模型
torch.save(model.state_dict(), "multimodal_bert_with_attention.pth")