import mysql
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
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

# 1. 数据预处理
data_cleaned = data.dropna(subset=['user_id', 'resource_id', 'country_id', 'career_id'])
data_cleaned['user_id'] = data_cleaned['user_id'].astype(str)
data_cleaned['resource_id'] = data_cleaned['resource_id'].astype(str)

# 2. 数据编码
user_encoder = LabelEncoder()
resource_encoder = LabelEncoder()
data_cleaned['user_id_encoded'] = user_encoder.fit_transform(data_cleaned['user_id'])
data_cleaned['resource_id_encoded'] = resource_encoder.fit_transform(data_cleaned['resource_id'])

# Extract edges
edges = data_cleaned[['user_id_encoded', 'resource_id_encoded']].values.T

# 3. 创建图数据
num_users = data_cleaned['user_id_encoded'].nunique()
num_resources = data_cleaned['resource_id_encoded'].nunique()
x = torch.ones(num_users + num_resources, 1)
graph_data = Data(x=x, edge_index=torch.tensor(edges, dtype=torch.long))

# Create labels
labels = torch.zeros(num_users, num_resources)
for _, row in data_cleaned.iterrows():
    labels[row['user_id_encoded'], row['resource_id_encoded']] = 1

# 4. 构建图神经网络
# Split the data into training and validation sets
train_indices, val_indices = train_test_split(range(num_users), test_size=0.1, random_state=42)

train_labels = labels[train_indices]
val_labels = labels[val_indices]
hidden_channels = 128


class EnhancedGraphRecSys(torch.nn.Module):
    def __init__(self, num_users, num_resources, hidden_channels):
        super(EnhancedGraphRecSys, self).__init__()
        self.conv1 = GCNConv(1, hidden_channels)  # Adjusted input features to 1
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.user_out = torch.nn.Linear(hidden_channels, num_resources)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        return self.user_out(x[:num_users])
def train_epoch():
    model.train()
    optimizer.zero_grad()
    out = model(graph_data.x, graph_data.edge_index)
    loss = criterion(out, labels)
    loss.backward()
    optimizer.step()
    return loss.item()
def validate():
    model.eval()
    with torch.no_grad():
        out = model(graph_data.x, graph_data.edge_index)
        loss = criterion(out[val_indices], val_labels)
    return loss.item()

# Training parameters
model = EnhancedGraphRecSys(num_users, num_resources, hidden_channels=128)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
scheduler = StepLR(optimizer, step_size=10, gamma=0.7)
criterion = torch.nn.BCEWithLogitsLoss()

# Enhanced training loop with early stopping
epochs = 20 #含义：这是您希望模型在整个数据集上进行训练的次数。
patience = 10 #这是一个早停参数。它指定了当验证损失不再改进时，我们应该等待多少个连续的epoch。
best_val_loss = float('inf') #含义：这是迄今为止观察到的最低验证损失。
counter = 0 #含义：这是一个计数器，用于跟踪自上次观察到最佳验证损失以来经过了多少个epoch。

for epoch in tqdm(range(epochs)):
    train_loss = train_epoch()
    val_loss = validate()

    scheduler.step()

    print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
    else:
        counter += 1
        if counter == patience:
            print("Early stopping triggered.")
            break

# 5. 进行推荐
model.eval()
with torch.no_grad():
    recommendations = model(graph_data.x, graph_data.edge_index)

# For demonstration, get top 10 recommendations for 10 users
top_k = 10
sample_users = range(1000)

# Extract recommendations for the sample users and map back to original user IDs
recommended_resources_dict = {}

for user_encoded in sample_users:
    user_original = user_encoder.inverse_transform([user_encoded])[0]
    _, top_resources_encoded = recommendations[user_encoded].topk(top_k)
    top_resources_original = resource_encoder.inverse_transform(top_resources_encoded.tolist())
    recommended_resources_dict[user_original] = top_resources_original

# Convert to DataFrame and save to CSV
recommendations_df = pd.DataFrame.from_dict(recommended_resources_dict, orient='index').reset_index()
recommendations_df.columns = ['user_id'] + [f'recommended_resource_{i+1}' for i in range(top_k)]
recommendations_df.to_csv('/Users/jiaqi_zheng/Desktop/Coding/python/Recsys_git/Recsys_work/resource/recommendationsGraphTest.csv', index=False)
