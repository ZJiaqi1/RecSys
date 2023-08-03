import mysql
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# 协同过滤算法根据一个人的id推荐20个title

db = mysql.connect()
# user分析
sql='''
select * from dl_user_resources_train
'''
data = pd.read_sql(sql,db)
data.head()
data.isnull().sum()

# Create a user-item matrix
user_u = list(sorted(data.user_id.unique()))
item_u = list(sorted(data.resource_id.unique()))

row = data.user_id.astype(pd.api.types.CategoricalDtype(categories=user_u)).cat.codes
# Map each item's id to an integer index
col = data.resource_id.astype(pd.api.types.CategoricalDtype(categories=item_u)).cat.codes
# We have only positive interactions (i.e., interactions = 1)
data_sparse = csr_matrix((len(data)*[1], (row, col)), shape=(len(user_u), len(item_u)))
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
        similar_items = model_knn.kneighbors(data_sparse[:,item].T, return_distance=False)
        # Get the indices of the items.
        similar_items_indices = similar_items.flatten().tolist()
        items_to_recommend.update(similar_items_indices)

    # Remove the items that the user has already interacted with.
    items_to_recommend.difference_update(items_interacted)

    # If there are too many items, just recommend the top ones.
    if len(items_to_recommend) > n_recommendations:
        items_to_recommend = list(items_to_recommend)[:n_recommendations]

    return items_to_recommend

# Generate recommendations for the first 10 users
user_indices = list(range(10))
n_recommendations = 10
recommendations = {}

for user_index in user_indices:
    recommendations[user_index] = generate_recommendations(user_index, data_sparse, model_knn, n_recommendations)

recommendations

# Map the item indices to their original IDs
recommendations_ids = {}

# Create a dictionary for mapping resource_id to title
resource_id_to_title = pd.Series(data.meta_title.values,index=data.resource_id).to_dict()

# Map the item indices to their original IDs and title
recommendations_ids_titles = {}

for user_id, recommended_items_ids in recommendations_ids.items():
    recommended_items_titles = [resource_id_to_title[item_id] for item_id in recommended_items_ids]
    recommendations_ids_titles[user_id] = recommended_items_titles
print(recommendations_ids_titles)
for user_index, recommended_items in recommendations.items():
    recommended_items_ids = [item_u[item_index] for item_index in recommended_items]
    recommendations_ids[user_u[user_index]] = recommended_items_ids
print(recommendations_ids)

def recommend_for_user(user_id, user_u, item_u, data_sparse, model_knn, n_recommendations):
    # Map the user_id to the user index
    user_index = user_u.index(user_id)

    # Generate recommendations for the user
    recommended_items = generate_recommendations(user_index, data_sparse, model_knn, n_recommendations)

    # Map the item indices to their original IDs and titles
    recommended_items_ids = [item_u[item_index] for item_index in recommended_items]
    recommended_items_titles = [resource_id_to_title[item_id] for item_id in recommended_items_ids]

    return recommended_items_ids, recommended_items_titles

# Test the function with a user_id
user_id = 'c172cce0bb3f5984f0e2777794082ea9'
print(recommend_for_user(user_id, user_u, item_u, data_sparse, model_knn, n_recommendations=20))