import pandas as pd
import mysql
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate
from keras.optimizers import Adam

# 神经协同过滤系统
# Load the dataset
db = mysql.connect()
sql='''
select * from dl_user_resources_train
'''
data = pd.read_sql(sql,db)
# Check for missing values
missing_values = data.isnull().sum()
missing_values

# Select relevant columns
data_ncf = data[['user_id', 'resource_id']]

# Initialize label encoders
user_encoder = LabelEncoder()
resource_encoder = LabelEncoder()

# Fit and transform columns
data_ncf['user_id'] = user_encoder.fit_transform(data_ncf['user_id'])
data_ncf['resource_id'] = resource_encoder.fit_transform(data_ncf['resource_id'])
data_ncf.head()
# Split the data into training and testing sets
train, test = train_test_split(data_ncf, test_size=0.2, random_state=42)
train.shape, test.shape
# Hyperparameters
embedding_size = 50
# Get the number of unique entities
n_users = len(data_ncf['user_id'].unique())
n_resources = len(data_ncf['resource_id'].unique())
# User model
user_input = Input(shape=[1])
user_embedding = Embedding(n_users, embedding_size, embeddings_initializer='he_normal',
                           embeddings_regularizer='l2', name='user_embedding')(user_input)
user_vector = Flatten()(user_embedding)
# Resource model
resource_input = Input(shape=[1])
resource_embedding = Embedding(n_resources, embedding_size, embeddings_initializer='he_normal',
                               embeddings_regularizer='l2', name='resource_embedding')(resource_input)
resource_vector = Flatten()(resource_embedding)
# Merge user and resource vectors
merged = Concatenate()([user_vector, resource_vector])
# Dense layers
dense_1 = Dense(128, activation='relu')(merged)
dense_2 = Dense(64, activation='relu')(dense_1)
# Output layer
output = Dense(1)(dense_2)
# Compile the model
model = Model([user_input, resource_input], output)
model.compile(loss='mean_squared_error', optimizer=Adam())
# Add implicit feedback
train['rating'] = 1
# Train the model
model.fit([train.user_id, train.resource_id], train.rating, epochs=10, verbose=1)
# Get the trained user embeddings
user_embeddings = model.get_layer('user_embedding').get_weights()[0]
# Get the trained resource embeddings
resource_embeddings = model.get_layer('resource_embedding').get_weights()[0]
# Create a dictionary to hold the recommendations
recommendations = {}
for user_index in range(n_users):  # Assuming tqdm has been removed for simplicity
    # Get the original user_id from the user_index
    user_id = user_encoder.inverse_transform([user_index])[0]
    # Compute the dot product between the user embeddings and resource embeddings
    scores = np.dot(user_embeddings[user_index], resource_embeddings.T)
    # Get the top 10 resource indices
    resource_indices = np.argsort(scores)[::-1][:10]
    # Convert resource indices back to original IDs
    resource_ids = resource_encoder.inverse_transform(resource_indices)
    # Add the recommendations to the dictionary
    recommendations[user_id] = list(resource_ids)
# Convert the recommendations dictionary to a format suitable for CSV
csv_data = []
for user, resource_ids in recommendations.items():
    row = {'user_id': user}
    for i, resource_id in enumerate(resource_ids, 1):
        row[f'resource_id_{i}'] = resource_id
    csv_data.append(row)

# Convert the data to a pandas DataFrame
df = pd.DataFrame(csv_data)

# Write the DataFrame to a CSV file
output_path = '/Users/jiaqi_zheng/Desktop/Coding/python/Recsys_git/Recsys_work/resource/neuralCollaborativeFiltering.csv'
df.to_csv(output_path, index=False)
