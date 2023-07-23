from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import mysql

# ----------------------------------------
# 这个部分用于测试对定义协同过滤算法
# ----------------------------------------

# Load the dataset
db = mysql.connect()
sql='''
select * from dl_user_resources_train limit 0,8000
'''
data = pd.read_sql(sql,db)
# Display the first few rows
data.head()
# Count the unique values in each column
unique_counts = data.nunique()
unique_counts
# Check for missing values
missing_values = data.isnull().sum()
missing_values
# Get the top 100 users who accessed the most resources
top_users = data['user_id'].value_counts().head(100)
top_users
# Get the top 20 countries with the most active users
top_countries = data['country_id'].value_counts().head(20)
top_countries
# Get the top 100 most accessed resources
top_resources = data['resource_id'].value_counts().head(100)
top_resources

# Create a user-resource matrix
user_resource_matrix = data.groupby(['user_id', 'resource_id']).size().unstack(fill_value=0)

# Display the first few rows of the matrix
user_resource_matrix.head()
# Select the first 10 users for demonstration
small_user_resource_matrix = user_resource_matrix.iloc[:100]
# Calculate the cosine similarity between users
user_similarity = cosine_similarity(small_user_resource_matrix)
# Convert the results to a DataFrame for better readability
user_similarity_df = pd.DataFrame(user_similarity, index=small_user_resource_matrix.index, columns=small_user_resource_matrix.index)
# Display the DataFrame
user_similarity_df


def recommend_resources(user, similarity_matrix, user_resource_matrix, k=3):
    # Get the top K similar users to the target user
    similar_users = similarity_matrix[user].nlargest(k + 1).index[1:]

    # Get the resources visited by these similar users
    visited_resources = user_resource_matrix.loc[similar_users].sum()

    # Remove the resources already visited by the target user
    target_user_resources = user_resource_matrix.loc[user]
    recommended_resources = visited_resources[target_user_resources == 0]

    # Sort the recommended resources by the number of visits by the similar users
    recommended_resources = recommended_resources.sort_values(ascending=False)

    return recommended_resources


# Recommend resources for the first user
first_user = small_user_resource_matrix.index[5]
recommended_resources = recommend_resources(first_user, user_similarity_df, small_user_resource_matrix)
recommended_resources