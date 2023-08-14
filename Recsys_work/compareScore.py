import pandas as pd

# 读取两个CSV文件
df1 = pd.read_csv('/Users/jiaqi_zheng/Desktop/Coding/python/Recsys_git/Recsys_work/resource/recommendationsGraphTest.csv')
df2 = pd.read_csv('/Users/jiaqi_zheng/Desktop/Coding/python/Recsys_git/Recsys_work/resource/dl_hashed_522018.csv')

# 创建一个字典来存储第一个CSV中的user_id及其推荐资源
recommendations = {}
for index, row in df1.iterrows():
    user_id = row['user_id']
    recommended_resources = [row[f'recommended_resource_{i}'] for i in range(1, 11)]
    recommendations[user_id] = recommended_resources

# 创建一个列表来存储分数
scores = []

# 遍历第二个CSV，查找匹配的user_id，然后为resource_id分配分数
for index, row in df2.iterrows():
    user_id = row['user_id']
    resource_id = row['resource_id']

    # 如果user_id在第一个CSV中
    if user_id in recommendations:
        # 检查resource_id的位置，并分配相应的分数
        if resource_id in recommendations[user_id]:
            position = recommendations[user_id].index(resource_id) + 1
            score = 11 - position  # 1st position gets 10, 2nd gets 9, ... 10th gets 1
        else:
            score = 0
        scores.append({'user_id': user_id, 'score': score})

# 创建一个DataFrame并保存为CSV
result_df = pd.DataFrame(scores)
result_df.to_csv('/Users/jiaqi_zheng/Desktop/Coding/python/Recsys_git/Recsys_work/resource/output_scores.csv', index=False)
