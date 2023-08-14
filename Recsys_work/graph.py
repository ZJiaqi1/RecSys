import networkx as nx
import pandas as pd
import numpy as np
from collections import defaultdict
import csv
from tqdm import tqdm
import torch
import mysql
from node2vec import Node2Vec
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#图神经网络推荐系统为每一个用户提供推荐结果
# Load the dataset
db = mysql.connect()
sql='''
select * from dl_user_resources
'''
data = pd.read_sql(sql,db)

# Handle missing values
data['country_id'].fillna(-1, inplace=True)
data['career_id'].fillna(-1, inplace=True)
data['meta_title'].fillna('', inplace=True)
data['mata_description'].fillna('', inplace=True)

# Build the graph
G = nx.Graph()
user_nodes = data['user_id'].unique().tolist()
resource_nodes = data['resource_id'].unique().tolist()
G.add_nodes_from(user_nodes, bipartite=0)
G.add_nodes_from(resource_nodes, bipartite=1)
edges = [(row['user_id'], row['resource_id']) for _, row in data.iterrows()]
G.add_edges_from(edges)

# Generate node embeddings using Node2Vec
node2vec = Node2Vec(G, dimensions=128, walk_length=80, num_walks=300, workers=8)
model = node2vec.fit(window=10, min_count=1)
node_embeddings = {node: model.wv[node] for node in G.nodes()}

# Separate user and resource nodes
user_nodes = [node for node, data in G.nodes(data=True) if data['bipartite'] == 0]
resource_nodes = [node for node, data in G.nodes(data=True) if data['bipartite'] == 1]
# Create a matrix of user embeddings
user_embeddings = np.array([node_embeddings[user] for user in user_nodes])
# Compute cosine similarity between users
user_similarity = cosine_similarity(user_embeddings)
# Get the top 10 similar users for each user
top_similar_users = np.argsort(user_similarity, axis=1)[:, -10:-1]

# Generate recommendations
recommendations = defaultdict(list)
for idx, user in tqdm(enumerate(user_nodes), desc="Users", unit="user"):
    similar_users_resources = set()
    for similar_user_idx in top_similar_users[idx]:
        similar_user = user_nodes[similar_user_idx]
        similar_users_resources.update(
            [neighbor for neighbor in G.neighbors(similar_user) if neighbor in resource_nodes])

    user_resources = set(G.neighbors(user))
    recommended_resources = similar_users_resources - user_resources
    recommendations[user] = list(recommended_resources)[:10]

# Specify the filename
filename = '/Users/jiaqi_zheng/Desktop/Coding/python/Recsys_git/Recsys_work/resource/recommendationsGraph.csv'

# Save recommendations to a CSV file
csv_filename = 'recommendations.csv'
with open(csv_filename, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    header = ['User'] + [f'Recommended Resource {i}' for i in range(1, 11)]
    csv_writer.writerow(header)
    for user, recommended_resources in recommendations.items():
        row = [user] + recommended_resources
        csv_writer.writerow(row)

print(f"Recommendations saved to {csv_filename}")
