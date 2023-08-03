import pandas as pd
import mysql
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import MinMaxScaler

# ----------------------------------------
# 这个部分用于测试对定义对于单一用户的基于内容的推荐方法
# 最后可以用一个用户的id来测试，可以展示相关度最高的10个资源
# ----------------------------------------

# Load the dataset
db = mysql.connect()
sql='''
select * from dl_user_resources_train
'''
data = pd.read_sql(sql,db)

# Re-define the TF-IDF vectorizer and the scaler
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
scaler = MinMaxScaler()

content = data['content'] = data['title'].fillna('') + ' ' + data['meta_title'].fillna('') + ' ' + data['mata_description'].fillna('')

# Compute TF-IDF matrix
tfidf_matrix = tfidf_vectorizer.fit_transform(content)

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Construct a reverse map of indices and resource titles
indices = pd.Series(data.index, index=data['resource_id']).drop_duplicates()

# Normalize the cosine similarity scores to be between 0 and 1
cosine_sim = scaler.fit_transform(cosine_sim)

cosine_sim.shape, indices.head()

def get_recommendations(resource_id, cosine_sim=cosine_sim, data=data, indices=indices):
    # Get the index of the resource that matches the resource_id 获取与resource_id匹配的资源的索引
    idx = indices[resource_id]
    # Get the pairwsie similarity scores of all resources with that resource 获取所有资源与该资源的pairwsie相似度分数
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort the resources based on the similarity scores 根据相似度分数对资源进行排序
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores of the 10 most similar resources 获取 10 个最相似资源的分数
    sim_scores = sim_scores[1:11]
    # Get the resource indices
    resource_indices = [i[0] for i in sim_scores]
    # Return the top 10 most similar resources
    return data['title'].iloc[resource_indices]

# Test the function with a random resource_id
#get_recommendations('test')

def get_recommendedResults(user_id):
    sql = '''
    select resource_id from dl_user_resources_test where user_id = '{}' order by datetime desc LIMIT 1
    '''.format(user_id)
    resourceData = pd.read_sql(sql, db).iloc[0, 0]
    return get_recommendations(resourceData)