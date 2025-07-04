import pandas as pd
import joblib
import re
from gensim.parsing.preprocessing import preprocess_string, strip_punctuation, strip_numeric, remove_stopwords, strip_short

# 预处理函数
CUSTOM_FILTERS = [lambda x: x.lower(), strip_punctuation, strip_numeric, remove_stopwords, strip_short]
def clean_text(text):
    return re.sub(r"http\S+", "", text)

def preprocess_for_lda(text):
    return preprocess_string(clean_text(text), CUSTOM_FILTERS)

# 加载原始数据和模型
df = pd.read_excel("task1_results.xlsx")
lda_model = joblib.load("lda_model.joblib")
dictionary = joblib.load("lda_dictionary.joblib")

# 生成每条文本的主题编号（选择最大概率主题）
topics = []
for text in df["text"].astype(str):
    bow = dictionary.doc2bow(preprocess_for_lda(text))
    topic_distribution = lda_model.get_document_topics(bow)
    top_topic = max(topic_distribution, key=lambda x: x[1])[0] if topic_distribution else -1
    topics.append(top_topic)

# 添加新列并保存
df["主题预测"] = topics
df.to_excel("task1_results_with_topic.xlsx", index=False)
print("✅ 已添加主题预测列，保存为 task1_results_with_topic.xlsx")
