import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from sklearn.model_selection import train_test_split
import numpy as np
import mysql

# Load the dataset
db = mysql.connect()
sql = '''
select * from dl_hash
'''
data = pd.read_sql(sql,db)
sqlTrain='''
select * from train_set
'''
data_train = pd.read_sql(sqlTrain,db)
sqlTest = '''
select * from test_set
'''
data_test = pd.read_sql(sqlTest,db)

# Basic information about the dataset
info = {
    'Number of rows': data_train.shape[0],
    'Number of columns': data_train.shape[1],
    'Number of unique user_ids': data_train['user_id'].nunique(),
    'Number of unique resource_ids': data_train['resource_id'].nunique(),
    'First date': data_train['datetime'].min(),
    'Last date': data_train['datetime'].max(),
}

print(info)
info = {
    'Number of rows': data_test.shape[0],
    'Number of columns': data_test.shape[1],
    'Number of unique user_ids': data_test['user_id'].nunique(),
    'Number of unique resource_ids': data_test['resource_id'].nunique(),
    'First date': data_test['datetime'].min(),
    'Last date': data_test['datetime'].max(),
}

print(info)

# Convert datetime to datetime object
data['datetime'] = pd.to_datetime(data['datetime'])
data['datetime'] = pd.to_datetime(data['datetime'])

# User access count
user_access_count = data['user_id'].value_counts()

# Resource access count
resource_access_count = data['resource_id'].value_counts()

# Access count per hour
data['hour'] = data['datetime'].dt.hour
hourly_access_count = data['hour'].value_counts().sort_index()

# Plot user access count
plt.figure(figsize=(14, 6))
plt.hist(user_access_count, bins=100, log=True, color='skyblue', edgecolor='black')
plt.title('Distribution of the number of accesses per user')
plt.xlabel('Number of accesses')
plt.ylabel('Number of users (log scale)')
plt.show()

# Plot resource access count
plt.figure(figsize=(14, 6))
plt.hist(resource_access_count, bins=100, log=True, color='skyblue', edgecolor='black')
plt.title('Distribution of the number of accesses per resource')
plt.xlabel('Number of accesses')
plt.ylabel('Number of resources (log scale)')
plt.show()

# Plot hourly access count
plt.figure(figsize=(14, 6))
plt.bar(hourly_access_count.index, hourly_access_count.values, color='skyblue', edgecolor='black')
plt.title('Distribution of access count per hour')
plt.xlabel('Hour of the day')
plt.ylabel('Number of accesses')
plt.show()


# Due to limited computational resources, we will only use a subset of the data
# data_small = data_train.sample(frac=0.01, random_state=1)

# Convert user_id and resource_id to categorical variables
data_train['user_id'] = data_train['user_id'].astype('category')
data_train['resource_id'] = data_train['resource_id'].astype('category')
data_test['user_id'] = data_test['user_id'].astype('category')
data_test['resource_id'] = data_test['resource_id'].astype('category')
data['user_id'] = data['user_id'].astype('category')
data['resource_id'] = data['resource_id'].astype('category')
# Create a user-resource matrix
user_resource_matrix = sparse.coo_matrix((np.ones(len(data)),
                                          (data['user_id'].cat.codes,
                                           data['resource_id'].cat.codes)))

# Compute the similarity matrix
similarity_matrix = cosine_similarity(user_resource_matrix)

# Split the data into training set and test set
# train_data, test_data = train_test_split(data_small, test_size=0.2, random_state=1)

similarity_matrix.shape, data_train.shape, data_test.shape

def recommend(user_id, similarity_matrix, data, top_k=5):
    """
    为给定用户推荐资源
     参数：
     user_id：目标用户的id。
     相似度矩阵：用户-用户相似度矩阵。
     data：用户资源交互数据。
     top_k：推荐的资源数量。
     返回：
     推荐的资源 ID 列表。
    Recommend resources for a given user.
    Parameters:
    user_id: The id of the target user.
    similarity_matrix: The user-user similarity matrix.
    data: The user-resource interaction data.
    top_k: The number of resources to recommend.
    Returns:
    A list of recommended resource ids.
    """
    user_idx = data['user_id'].cat.categories.get_loc(user_id)
    similar_users = np.argsort(similarity_matrix[user_idx])[::-1][1:top_k+1]
    similar_users_ids = data['user_id'].cat.categories[similar_users]

    rec_resources = data[data['user_id'].isin(similar_users_ids)]['resource_id']
    rec_resources = rec_resources.value_counts().index[:top_k]

    return rec_resources.tolist()


# Test the recommend function
test_user = data_test['user_id'].iloc[0]
rec_resources = recommend(test_user, similarity_matrix, data_test)
rec_resources

# Evaluate the recommendation system on the test set
hits = 0
total = 0

for user_id in data_test['user_id'].unique():
    actual_resources = data_test[data_test['user_id'] == user_id]['resource_id'].tolist()
    rec_resources = recommend(user_id, similarity_matrix, data_test)
    hits += len(set(actual_resources) & set(rec_resources))
    total += len(rec_resources)

accuracy1 = hits / total
#协同过滤推荐系统结果
accuracy1


# Compute the popularity of each resource
resource_popularity = data_test['resource_id'].value_counts()

# Select the top 5 most popular resources as the recommendations
top_resources = resource_popularity.index[:5].tolist()

# Evaluate the recommendation system on the test set
hits = 0
total = 0

for user_id in data_test['user_id'].unique():
    actual_resources = data_test[data_test['user_id'] == user_id]['resource_id'].tolist()
    hits += len(set(actual_resources) & set(top_resources))
    total += len(top_resources)

accuracy2 = hits / total
#基于流行度的推荐系统结果
accuracy2

# Define the time slots
time_slots = {0: 'morning', 1: 'morning', 2: 'morning', 3: 'morning', 4: 'morning', 5: 'morning',
              6: 'forenoon', 7: 'forenoon', 8: 'forenoon', 9: 'forenoon', 10: 'forenoon', 11: 'forenoon',
              12: 'afternoon', 13: 'afternoon', 14: 'afternoon', 15: 'afternoon', 16: 'afternoon', 17: 'afternoon',
              18: 'evening', 19: 'evening', 20: 'evening', 21: 'evening', 22: 'evening', 23: 'evening'}

# Compute the popularity of each resource in each time slot
data_test['time_slot'] = data_test['hour'].map(time_slots)
resource_popularity = data_test.groupby('time_slot')['resource_id'].value_counts()

# Define a function to recommend resources based on time
def recommend_time(user_id, time_slot, resource_popularity, top_k=5):
    """
    Recommend resources for a given user based on time.

    Parameters:
    user_id: The id of the target user.
    time_slot: The time slot.
    resource_popularity: The popularity of each resource in each time slot.
    top_k: The number of resources to recommend.

    Returns:
    A list of recommended resource ids.
    """
    top_resources = resource_popularity.loc[time_slot].index[:top_k].tolist()
    return top_resources

# Add 'time_slot' to the test data
data_test['time_slot'] = data_test['hour'].map(time_slots)

# Test the recommend function again
test_time_slot = data_test[data_test['user_id'] == test_user]['time_slot'].iloc[0]
rec_resources = recommend_time(test_user, test_time_slot, resource_popularity)
rec_resources

# Evaluate the recommendation system on the test set
hits = 0
total = 0

for user_id in data_test['user_id'].unique():
    actual_resources = data_test[data_test['user_id'] == user_id]['resource_id'].tolist()
    time_slot = data_test[data_test['user_id'] == user_id]['time_slot'].iloc[0]
    rec_resources = recommend_time(user_id, time_slot, resource_popularity)
    hits += len(set(actual_resources) & set(rec_resources))
    total += len(rec_resources)

accuracy3 = hits / total
#基于时间的推荐系统结果
accuracy3
