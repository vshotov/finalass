# -*- coding: utf-8 -*-
# 使用前10条推文进行 LDA 主题建模（无 nltk，去除网址）

import re
import gensim
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from gensim import corpora, models
from gensim.parsing.preprocessing import preprocess_string, strip_punctuation, strip_numeric, strip_short, remove_stopwords

import tempfile
import os
os.environ['JOBLIB_TEMP_FOLDER'] = tempfile.mkdtemp(dir='D:\\temp')


# ✅ 设置字体，避免 matplotlib 中文乱码（如无 SimHei 可替换为 Microsoft YaHei）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ✅ 文本预处理函数（无 nltk，去除网址）
CUSTOM_FILTERS = [lambda x: x.lower(), strip_punctuation, strip_numeric, remove_stopwords, strip_short]
def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # 去除网址
    return text

def preprocess(texts):
    return [preprocess_string(clean_text(doc), CUSTOM_FILTERS) for doc in texts]

# ✅ 前10条推文（数据直接写入）
data = [
    "Don't need feds to solve the #bostonbombing when we have #4chan!! http://t.co/eXQTPZqqbG",
    "PIC: Comparison of #Boston suspect Sunil Tripathi's FBI-released images/video and his MISSING poster http://t.co/EhV3ODxrJf You decide.",
    "I'm not completely convinced that it's this Sunil Tripathi fellow—http://t.co/ZjWqOsRfjB",
    "Brutal lo que se puede conseguir en colaboración. #4Chan analizando fotos de la maratón de #Boston atando cabos... http://t.co/4eq43HFicK",
    "4chan and the bombing. just throwing it out there: http://t.co/dIySO7lXQm http://t.co/NxBi4tW8bQ",
    "4chan thinks they found pictures of the bomber  -  http://t.co/0y3w24l2q8",
    "Ola ke ase, investigando las bombas de Boston o ke ase? #4chan http://t.co/7A4IAbVmM0",
    "4chan ThinkTank - Imgur http://t.co/hQt2fhxE48",
    "@DLoesch have you seen this? Bomber #2  looks like missing student Sunil Tripathi.  http://t.co/1phQpB0aMT  http://t.co/X9pqMtnoW5",
    "da 4chan think tank BOSTON  http://t.co/0ZbKMA5z0D  #photos #suspects #bombing #marathon"
]

# ✅ 处理文本
cleaned_data = preprocess(data)

# ✅ 构建字典与语料库
dictionary = corpora.Dictionary(cleaned_data)
corpus = [dictionary.doc2bow(text) for text in cleaned_data]

# ✅ 训练 LDA 模型（3个主题）
lda_model = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=3, random_state=42, passes=10)

# ✅ 打印每个主题关键词
print("\n🔍 每个主题的关键词：")
for idx, topic in lda_model.print_topics(num_words=5):
    print(f"主题 {idx}: {topic}")

# ✅ 生成词云图
for i in range(3):
    plt.figure()
    word_freqs = dict(lda_model.show_topic(i, topn=30))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freqs)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"主题 {i} 词云")
    plt.tight_layout()
    plt.savefig(f"topic_{i}_wordcloud.png")

# ✅ 生成 pyLDAvis 可视化页面
vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
pyLDAvis.save_html(vis, 'lda_vis.html')
print("\n✅ 可视化已保存为 lda_vis.html，请用浏览器打开查看。")

import joblib

joblib.dump(lda_model, "lda_model.joblib")
joblib.dump(dictionary, "lda_dictionary.joblib")

