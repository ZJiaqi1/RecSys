import mysql
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

# ----------------------------------------
# 这个部分用于评估基于内容的推荐方法
# ----------------------------------------

db = mysql.connect()
sql_train='''
select * from dl_user_resources_train
'''
data_train = pd.read_sql(sql_train,db)
# Display the first few rows
#print(data_train.head())
sql_test='''
select * from dl_user_resources_test
'''
data_test = pd.read_sql(sql_test,db)
# Display the first few rows
#print(data_test.head())

# 首先，我们找出每个用户最常使用的资源。我们可以通过计算每个用户与每个资源之间的交互次数来实现这一目标，然后选择交云最多的资源作为预测结果。
# Group the training data by user_id and resource_id, and count the number of interactions for each pair.
# Then, for each user, select the resource with the most interactions as the prediction.
train_grouped = data_train.groupby(['user_id', 'resource_id']).size().reset_index(name='counts')
predictions = train_grouped.sort_values('counts', ascending=False).drop_duplicates(['user_id'])
# 生成了每个用户最常使用的资源作为预测结果
# We only need the user_id and resource_id columns for the predictions.
predictions = predictions[['user_id', 'resource_id']]
predictions.head()

# Create a dataframe of all user-resource pairs in the test set
test_pairs = data_test[['user_id', 'resource_id']]

# Create a label for each user-resource pair in the test set
# Label is 1 if the user has interacted with the resource, 0 otherwise
test_pairs['label'] = 1

# Merge the test labels with the predictions
# Predicted label is 1 if the resource was predicted for the user, 0 otherwise
results = test_pairs.merge(predictions, how='left', on=['user_id', 'resource_id'], indicator=True)
results['predicted'] = results['_merge'] == 'both'
results['predicted'] = results['predicted'].astype(int)

# Calculate precision, recall, and F1 score
precision = precision_score(results['label'], results['predicted'])
recall = recall_score(results['label'], results['predicted'])
f1 = f1_score(results['label'], results['predicted'])
#精确度（Precision）
#召回率（Recall）
#F1分数（F1 Score）
print(precision)
print(recall)
print(f1)


