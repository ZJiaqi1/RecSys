import mysql
import torch
import csv
from collections import defaultdict
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx
import networkx as nx
import pandas as pd
import psutil
from tqdm import tqdm

# Determine if GPU acceleration is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 图神经网络推荐系统为每一个用户提供推荐结果
# Load the dataset
db = mysql.connect()
sql = '''
select * from dl_user_resources
'''
data = pd.read_sql(sql, db)

# Create a graph
G = nx.Graph()

# Add nodes for users and resources
for user in data['user_id'].unique():
    G.add_node(user, type='user')

for resource in data['resource_id'].unique():
    G.add_node(resource, type='resource')

# Add edges between users and resources they have accessed
for _, row in data.iterrows():
    G.add_edge(row['user_id'], row['resource_id'])

# Convert NetworkX graph to PyTorch Geometric data format
pyg_data = from_networkx(G).to(device)

# Create labels: 1 for user-resource interactions, 0 for non-interactions
labels = [1 if edge in G.edges() else 0 for edge in G.edges()]
pyg_data.y = torch.tensor(labels, dtype=torch.float).to(device)
# Convert NetworkX graph to PyTorch Geometric data format
pyg_data = from_networkx(G)

# Add node features for all nodes (set as 1 for simplicity)
pyg_data.x = torch.ones(pyg_data.num_nodes, 1).to(device)

# Create labels: 1 for user-resource interactions, 0 for non-interactions
labels = [1 if edge in G.edges() else 0 for edge in G.edges()]
pyg_data.y = torch.tensor(labels, dtype=torch.float).to(device)


# Define a Graph Neural Network using PyTorch Geometric
class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(1, 128)  # First GCN layer: input feature size is 1, output feature size is 128
        self.conv2 = GCNConv(128, 128)  # Second GCN layer: input feature size is 128, output feature size is also 128

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x


# Initialize the model, loss function, and optimizer
model = GNN().to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# Train the model
def train(data):
    model.train()
    optimizer.zero_grad()
    node_embeddings = model(data)  # This will give embeddings for each node
    edge_embeddings = torch.cat([node_embeddings[data.edge_index[0]], node_embeddings[data.edge_index[1]]],
                                dim=1)  # Concatenate embeddings of source and target nodes for each edge

    # Use a simple linear layer to predict from edge embeddings
    pred_layer = torch.nn.Linear(2 * 128, 1).to(device)  # Assuming 128 is the embedding size
    predictions = torch.sigmoid(pred_layer(edge_embeddings))

    loss = criterion(predictions[data.train_mask], data.y[data.train_mask].unsqueeze(1))
    loss.backward()
    optimizer.step()
    return loss.item()


# Create labels only for actual edges: set all as 1 since all are interactions
pyg_data.y = torch.ones(pyg_data.edge_index.shape[1]).to(device)

# Assuming 80% of the edges for training and 20% for validation
num_edges = pyg_data.edge_index.shape[1]
train_mask = torch.zeros(num_edges, dtype=torch.bool)
train_mask[:int(0.8 * num_edges)] = 1
pyg_data.train_mask = train_mask

# Training loop
for epoch in tqdm(range(100), desc="Training", unit="epoch"):
    loss = train(pyg_data)
    # 获取当前进程的内存使用情况
    process = psutil.Process()
    mem_info = process.memory_info()
    rss_memory = mem_info.rss / (1024 ** 2)  # 获取Resident Set Size内存，单位为MB
    print(f"Epoch {epoch + 1}, Loss: {loss:.4f}, Memory used: {rss_memory:.2f} MB")
print("Training completed!")
model.eval()  # Set the model to evaluation mode

# Predict scores for all user-resource pairs
all_scores = model(pyg_data)

# Extract user and resource nodes
user_nodes = [node for node, data in G.nodes(data=True) if data['type'] == 'user']
resource_nodes = [node for node, data in G.nodes(data=True) if data['type'] == 'resource']

# Create a dictionary to store the top 5 recommended resources for each user
recommendations = {}

for user in tqdm(user_nodes, desc="Users", unit="user"):
    scores = []
    for resource in resource_nodes:
        if not G.has_edge(user, resource):  # Only consider resources not interacted with by the user
            nodes_list = list(G.nodes())
            # 获取用户和资源的索引
            user_index = nodes_list.index(user)
            resource_index = nodes_list.index(resource)
            # 从模型的输出中提取用户和资源的嵌入
            user_embedding = all_scores[user_index]
            resource_embedding = all_scores[resource_index]
            # 生成预测分数（这里我们简单地使用点积，但您可以使用更复杂的方法）
            score = torch.dot(user_embedding, resource_embedding)
            scores.append((resource, score.item()))
    # Sort resources by score and select the top 10
    top_resources = sorted(scores, key=lambda x: x[1], reverse=True)[:10]
    recommendations[user] = [resource[0] for resource in top_resources]

# Save the recommendations to a csv file
with open('/Users/jiaqi_zheng/Desktop/Coding/python/Recsys_git/Recsys_work/resource/recommendationsGraphTest.csv',
          'w') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(
        ['User', 'Recommended Resource 1', 'Recommended Resource 2', 'Recommended Resource 3', 'Recommended Resource 4',
         'Recommended Resource 5', 'Recommended Resource 6', 'Recommended Resource 7', 'Recommended Resource 8',
         'Recommended Resource 9', 'Recommended Resource 10'])
    for user, recommended_resources in recommendations.items():
        row = [user] + recommended_resources
        csv_writer.writerow(row)

print("Recommendations saved to 'recommendationsGraphTest.csv'")
