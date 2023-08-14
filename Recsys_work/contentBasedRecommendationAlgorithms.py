import mysql
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# 基于内容的推荐算法
# Load the dataset
db = mysql.connect()
sql='''
select * from dl_user_resources_train
'''
data = pd.read_sql(sql,db)

# 提供一个简单的停用词列表
simple_stop_words = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
    "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
    'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those',
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
    'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
    'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's',
    't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y'
}
def preprocess_text_without_lemmatization(text):
    """
    不进行词形还原的文本预处理函数:
    1. 转换为小写
    2. 移除非字母字符
    3. 分词并去除停用词
    4. 重新组合为字符串
    """
    if not isinstance(text, str):
        return ""

    # 转换为小写
    text = text.lower()

    # 移除非字母字符
    text = re.sub(r'[^a-z\s]', '', text)

    # 分词
    words = text.split()

    # 去除停用词
    words = [word for word in words if word not in simple_stop_words]

    # 重新组合为字符串
    return ' '.join(words)


# 对数据中的文本列进行预处理
data['title'] = data['title'].apply(preprocess_text_without_lemmatization)
data['meta_title'] = data['meta_title'].apply(preprocess_text_without_lemmatization)
data['mata_description'] = data['mata_description'].apply(preprocess_text_without_lemmatization)

# 显示处理后的前几行数据
data[['title', 'meta_title', 'mata_description']].head()

# 使用TF-IDF向量化器
vectorizer = TfidfVectorizer(max_features=5000)
data['combined_text'] = data['title'] + ' ' + data['meta_title'] + ' ' + data['mata_description']
tfidf_matrix = vectorizer.fit_transform(data['combined_text'])

tfidf_matrix.shape

from sklearn.metrics.pairwise import linear_kernel

# 选择前1000个独立的用户
selected_users = data['user_id'].drop_duplicates().head(4000).tolist()

# 为每个用户推荐资源
recommendations_for_first_10 = {}

for user in selected_users:
    # 获取该用户互动过的资源的索引
    user_indices = data[data['user_id'] == user].index.tolist()

    # 计算这些资源与其他资源的相似度
    cosine_similarities = sum(
        [linear_kernel(tfidf_matrix[index:index + 1], tfidf_matrix).flatten() for index in user_indices])

    # 获取相似度得分最高的资源
    similar_indices = cosine_similarities.argsort()[-(10 + len(user_indices)):][::-1]

    # 过滤掉用户已经互动过的资源
    recommended_indices = [i for i in similar_indices if i not in user_indices][:10]

    recommended_resources = data.iloc[recommended_indices]

    recommendations_for_first_10[user] = recommended_resources[['resource_id', 'title']].to_dict(orient='records')

recommendations_for_first_10
csv_data = []
for user, recommendations in recommendations_for_first_10.items():
    row = {'user': user}
    for i, rec in enumerate(recommendations, 1):
        row[f'resource_id_{i}'] = rec['resource_id']
        row[f'title_{i}'] = rec['title']
    csv_data.append(row)

# 创建一个DataFrame
df = pd.DataFrame(csv_data)

# 保存为CSV文件
output_path = "/Users/jiaqi_zheng/Desktop/Coding/python/Recsys_git/Recsys_work/resource/recommendationsCBRA.csv"
df.to_csv(output_path, index=False)








