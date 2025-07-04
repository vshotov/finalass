import pandas as pd
from tqdm import tqdm
import ollama

# ======== 设置 ========
MODEL_NAME = 'deepseek-r1:1.5b'  # 可换成 tinyllama / phi:2 以加速
DATA_FILE = 'posts_groundtruth.txt'
OUTPUT_FILE = 'task1_results.xlsx'
LIMIT_ROWS = None  # 设置为 None 则不限制条数
# ======================

# 加载数据
df = pd.read_csv(DATA_FILE, sep='\t')
news_data = df[['post_id', 'post_text', 'label']].rename(columns={'post_text': 'text', 'label': 'label'})
if LIMIT_ROWS:
    news_data = news_data.head(LIMIT_ROWS)

# 标签统一函数，统一成小写且仅 true/fake/unknown
def unify_label(label):
    label = str(label).lower()
    if label in ['true', '真实', '真']:
        return 'true'
    elif label in ['fake', 'false', '假']:
        return 'fake'
    else:
        return 'unknown'

news_data['label'] = news_data['label'].apply(unify_label)

# 模型调用函数
def call_model(prompt, retry=3):
    for _ in range(retry):
        try:
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}]
            )
            return response['message']['content']
        except Exception as e:
            print(f"❌ 模型调用失败：{e}")
    return "调用失败"

# 提取三个结果：初始预测、情感、联合判断（更健壮）
def extract_all(reply):
    lines = reply.strip().lower().splitlines()
    result = {'初始预测': 'unknown', '情感倾向': '未知', '情感辅助预测': 'unknown'}

    for line in lines:
        line = line.strip()
        # 解析初始判断
        if line.startswith('初始判断'):
            if 'true' in line:
                result['初始预测'] = 'true'
            elif 'false' in line:
                result['初始预测'] = 'fake'  # 把 false 映射为 fake
            elif 'fake' in line:
                result['初始预测'] = 'fake'
            elif 'unknown' in line:
                result['初始预测'] = 'unknown'
        # 解析情感倾向
        elif line.startswith('情感倾向'):
            if '正面' in line:
                result['情感倾向'] = '正面'
            elif '负面' in line:
                result['情感倾向'] = '负面'
            elif '中性' in line:
                result['情感倾向'] = '中性'
            else:
                result['情感倾向'] = '未知'
        # 解析联合判断
        elif line.startswith('联合判断'):
            if 'true' in line:
                result['情感辅助预测'] = 'true'
            elif 'false' in line:
                result['情感辅助预测'] = 'fake'  # false 映射为 fake
            elif 'fake' in line:
                result['情感辅助预测'] = 'fake'
            elif 'unknown' in line:
                result['情感辅助预测'] = 'unknown'
    return result

# 合并 prompt，一次完成全部任务，强制无解释输出
def analyze_all(text):
    prompt = (
        "你是一个信息抽取机器人。\n"
        "请完成以下任务：\n"
        "1. 判断新闻是真是假，只回答 true 或 fake\n"
        "2. 判断情感：正面、负面、中性\n"
        "3. 根据新闻和情感再次判断真假，只回答 true 或 fake\n\n"
        "请严格按照以下格式输出，且不要包含任何额外解释或推理内容：\n"
        "初始判断: true/fake\n"
        "情感倾向: 正面/负面/中性\n"
        "联合判断: true/fake\n\n"
        f"新闻内容如下：\n{text}"
    )
    reply = call_model(prompt)
    return extract_all(reply), reply

# 执行主循环
results = []
print(f"🚀 开始处理 {len(news_data)} 条新闻...\n")

for i, row in tqdm(news_data.iterrows(), total=len(news_data), desc="处理中"):
    text = row['text']
    result, raw_reply = analyze_all(text)

    # Debug 输出
    print(f"\n📰 第{i+1}条新闻：")
    #print(f"模型返回：\n{raw_reply}")
    print(f"→ 初始判断: {result['初始预测']}, 情感: {result['情感倾向']}, 联合判断: {result['情感辅助预测']}")

    results.append(result)

# 合并结果
for key in ['初始预测', '情感倾向', '情感辅助预测']:
    news_data[key] = [r[key] for r in results]

# 准确率计算，忽略 unknown
def calc_accuracy(pred_col):
    mask_known = news_data[pred_col].isin(['true', 'fake']) & news_data['label'].isin(['true', 'fake'])
    correct = (news_data.loc[mask_known, pred_col] == news_data.loc[mask_known, 'label'])
    acc = correct.sum() / len(correct) if len(correct) > 0 else 0

    fake_mask = news_data['label'] == 'fake'
    true_mask = news_data['label'] == 'true'

    acc_fake = (news_data.loc[fake_mask & mask_known, pred_col] == 'fake').sum() / fake_mask.sum() if fake_mask.sum() else 0
    acc_true = (news_data.loc[true_mask & mask_known, pred_col] == 'true').sum() / true_mask.sum() if true_mask.sum() else 0

    return acc, acc_fake, acc_true

acc1, acc1_fake, acc1_true = calc_accuracy('初始预测')
acc2, acc2_fake, acc2_true = calc_accuracy('情感辅助预测')

# 输出准确率
print("\n🎯 准确率统计：")
print(f"✅ 初始预测 Accuracy = {acc1:.4f} | 假新闻: {acc1_fake:.4f} | 真新闻: {acc1_true:.4f}")
print(f"✅ 联合判断 Accuracy = {acc2:.4f} | 假新闻: {acc2_fake:.4f} | 真新闻: {acc2_true:.4f}")

# 保存结果
news_data.to_excel(OUTPUT_FILE, index=False)
print(f"\n📁 所有结果已保存至：{OUTPUT_FILE}")
