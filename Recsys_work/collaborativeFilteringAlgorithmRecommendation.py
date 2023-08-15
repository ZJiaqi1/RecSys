import mysql
import pandas as pd
from tqdm import tqdm
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# 协同过滤算法根据一个人的id推荐10个推荐的资源

db = mysql.connect()
# user分析
sql = '''
select * from dl_user_resources_train
'''
data = pd.read_sql(sql, db)
data.head()
data.isnull().sum()

# Create a user-item matrix
user_u = list(sorted(data.user_id.unique()))
item_u = list(sorted(data.resource_id.unique()))

row = data.user_id.astype(pd.api.types.CategoricalDtype(categories=user_u)).cat.codes
# Map each item's id to an integer index
col = data.resource_id.astype(pd.api.types.CategoricalDtype(categories=item_u)).cat.codes
# We have only positive interactions (i.e., interactions = 1)
data_sparse = csr_matrix((len(data) * [1], (row, col)), shape=(len(user_u), len(item_u)))
data_sparse.shape
# Make an object for the NearestNeighbors Class.
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
# Fit the dataset
model_knn.fit(data_sparse.T)


def generate_recommendations(user_index, data_sparse, model_knn, n_recommendations):
    n_users, n_items = data_sparse.shape
    items_to_recommend = set()
    # Get the items that the user has interacted with.
    items_interacted = data_sparse[user_index].indices
    # Get similar items for each item the user has interacted with.
    for item in items_interacted:
        similar_items = model_knn.kneighbors(data_sparse[:, item].T, return_distance=False)
        # Get the indices of the items.
        similar_items_indices = similar_items.flatten().tolist()
        items_to_recommend.update(similar_items_indices)
    # Remove the items that the user has already interacted with.
    items_to_recommend.difference_update(items_interacted)
    # If there are too many items, just recommend the top ones.
    if len(items_to_recommend) > n_recommendations:
        items_to_recommend = list(items_to_recommend)[:n_recommendations]
    return items_to_recommend


# Generate recommendations for the first 4000 users
user_indices = list(range(4000))
n_recommendations = 10
recommendations_ids = {}  # This dictionary will hold the user_id and their recommended resource_ids

for user_index in tqdm(user_indices, desc="User Progress"):
    recommended_items = generate_recommendations(user_index, data_sparse, model_knn, n_recommendations)
    recommended_items_ids = [item_u[item_index] for item_index in recommended_items]
    recommendations_ids[user_u[user_index]] = recommended_items_ids
resource_id_to_title = pd.Series(data.meta_title.values, index=data.resource_id).to_dict()

csv_data = []
total_users = len(recommendations_ids)
for user_idx, (user_id, resource_ids) in tqdm(enumerate(recommendations_ids.items()), total=total_users,
                                              desc="CSV Construction"):
    row = {'user_id': user_id}
    for i, resource_id in enumerate(resource_ids, 1):
        row[f'resource_id_{i}'] = resource_id
        row[f'title_{i}'] = resource_id_to_title[resource_id]
    csv_data.append(row)
# Convert the data to a pandas DataFrame
df = pd.DataFrame(csv_data)
# Write the DataFrame to a CSV file
output_path = '/Users/jiaqi_zheng/Desktop/Coding/python/Recsys_git/Recsys_work/resource/collaborativeFilteringAlgorithmRecommendation.csv'  # Change this path to where you want to save the CSV
df.to_csv(output_path, index=False)