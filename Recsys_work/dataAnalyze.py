import mysql
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# 数据图表分析


db = mysql.connect()
# user分析
sqlUser='''
select * from user_hashed
'''
dataUser = pd.read_sql(sqlUser,db)
# General info of the dataset
dataUser.info()
# Descriptive statistics for numerical columns
dataUser.describe(include=[np.number])
# Plot histogram for country_id
plt.figure(figsize=(10, 5))
plt.hist(dataUser['country_id'].dropna(), bins=50, alpha=0.5, color='g')
plt.title('Histogram of country_id')
plt.xlabel('Country ID')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
# Plot histogram for career_id
plt.figure(figsize=(10, 5))
plt.hist(dataUser['career_id'].dropna(), bins=50, alpha=0.5, color='b')
plt.title('Histogram of career_')
plt.xlabel('Career ID')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# resource分析
sqlResource='''
select * from resources_hashed
'''
dataResource = pd.read_sql(sqlResource,db)
# Function to get most common elements
def get_most_common(column, num=10):
    # Flatten the list and count the frequency of each item
    count = Counter(column)
    # Get the most common items
    most_common = count.most_common(num)
    return most_common
dataResource['date_updated'].unique()
# Convert date columns to datetime format, ignore errors
dataResource['date_added'] = pd.to_datetime(dataResource['date_added'], errors='coerce')
dataResource['date_updated'] = pd.to_datetime(dataResource['date_updated'], errors='coerce')
# Get the most common elements in 'title', 'meta_title', 'meta_keywords', and 'meta_description'
most_common_title = get_most_common(dataResource['title'])
most_common_meta_title = get_most_common(dataResource['meta_title'].dropna())
most_common_meta_keywords = get_most_common(dataResource['meta_keywords'].dropna())
# most_common_meta_description = get_most_common(dataResource['meta_description'].dropna())

# Plot the distribution of 'date_added' and 'date_updated'
fig, ax = plt.subplots(2, 1, figsize=(15, 10))
sns.histplot(dataResource['date_added'].dt.year, ax=ax[0], kde=False, bins=range(dataResource['date_added'].dt.year.min(), dataResource['date_added'].dt.year.max()+1))
ax[0].set_title('Distribution of "date_added"')
sns.histplot(dataResource['date_updated'].dt.year.dropna(), ax=ax[1], kde=False, bins=range(int(dataResource['date_updated'].dt.year.min()), int(dataResource['date_updated'].dt.year.max()+1)))
ax[1].set_title('Distribution of "date_updated"')
plt.tight_layout()
plt.show()
most_common_title, most_common_meta_title

# 视图分析
sqlView='''
select * from dl_user_resources
'''
dataView = pd.read_sql(sqlView,db)
dataView.head()
dataView.info()
# Convert 'datetime' to datetime format
dataView['datetime'] = pd.to_datetime(dataView['datetime'])

# Extract the hour from 'datetime'
dataView['hour'] = dataView['datetime'].dt.hour
# Plot the distribution of user activity by hour 这个图表展示了用户活动的小时分布情况。我们可以看到，在一天中的某些特定时间（如早上的8点到9点，下午的13点到14点，以及晚上的19点到20点）用户的活动量更大。这些时间可能是用户上课或者工作的高峰期。
plt.figure(figsize=(12,6))
sns.countplot(x='hour', data=dataView, color='blue')
plt.title('Distribution of User Activity by Hour')
plt.xlabel('Hour')
plt.ylabel('Count')
plt.show()

# Plot the distribution of users by country (Top 10 countries)
plt.figure(figsize=(12,6))
dataView['country_id'].value_counts().head(10).plot(kind='bar', color='blue')
plt.title('Distribution of Users by Country (Top 10)')
plt.xlabel('Country ID')
plt.ylabel('Count')
plt.show()

# Plot the distribution of users by career (Top 10 careers)
plt.figure(figsize=(12,6))
dataView['career_id'].value_counts().head(10).plot(kind='bar', color='blue')
plt.title('Distribution of Users by Career (Top 10)')
plt.xlabel('Career ID')
plt.ylabel('Count')
plt.show()

# Plot the distribution of resources (Top 10 resources)
plt.figure(figsize=(12,6))
dataView['resource_id'].value_counts().head(10).plot(kind='bar', color='blue')
plt.title('Distribution of Resources (Top 10)')
plt.xlabel('Resource ID')
plt.ylabel('Count')
plt.show()

# Create a cross table for 'country_id' and 'career_id'
cross_table_country_career = pd.crosstab(dataView['country_id'], dataView['career_id'])

# Select the top 10 countries and careers
top_10_countries = dataView['country_id'].value_counts().head(10).index
top_10_careers = dataView['career_id'].value_counts().head(10).index
# Filter the data to include only the top 10 countries and careers
filtered_data = dataView[dataView['country_id'].isin(top_10_countries) & dataView['career_id'].isin(top_10_careers)]

# Create a cross table for the filtered data
cross_table_country_career_filtered = pd.crosstab(filtered_data['country_id'], filtered_data['career_id'])

# Plot the heatmap 这个热图展示了前10个国家和职业的用户数量。颜色越深，表示用户数量越多。
plt.figure(figsize=(12,6))
sns.heatmap(cross_table_country_career_filtered, cmap='Blues', annot=True, fmt='d')
plt.title('Heatmap of User Counts by Country and Career (Top 10)')
plt.xlabel('Career ID')
plt.ylabel('Country ID')
plt.show()

# Create a cross table for 'hour' and 'career_id' 用户活动时间和职业之间的关系
cross_table_hour_career = pd.crosstab(dataView['hour'], dataView['career_id'])

# Select the top 10 careers
cross_table_hour_career = cross_table_hour_career[top_10_careers]

# Plot the heatmap
plt.figure(figsize=(12,6))
sns.heatmap(cross_table_hour_career, cmap='Blues', annot=True, fmt='d')
plt.title('Heatmap of User Activity by Hour and Career (Top 10)')
plt.xlabel('Career ID')
plt.ylabel('Hour')
plt.show()

# Get the top 10 resources 用户所在国家和资源使用情况的关系
top_10_resources = dataView['resource_id'].value_counts().head(10).index

# Create a cross table for 'country_id' and 'resource_id'
cross_table_country_resource = pd.crosstab(dataView['country_id'], dataView['resource_id'])

# Select the top 10 countries and resources
cross_table_country_resource = cross_table_country_resource.loc[top_10_countries, top_10_resources]

# Plot the heatmap
plt.figure(figsize=(12,6))
sns.heatmap(cross_table_country_resource, cmap='Blues', annot=True, fmt='d')
plt.title('Heatmap of Resource Usage by Country and Resource (Top 10)')
plt.xlabel('Resource ID')
plt.ylabel('Country ID')
plt.show()

# Filter the data to include only the top 10 countries and careers看看用户活动时间、国家和职业的关系
filtered_data = dataView[dataView['country_id'].isin(top_10_countries) & dataView['career_id'].isin(top_10_careers)]

# Create a cross table for 'hour', 'country_id' and 'career_id'
cross_table_hour_country_career = filtered_data.groupby(['hour', 'country_id', 'career_id']).size().reset_index(name='count')

# Pivot the table for the heatmap
cross_table_hour_country_career_pivot = cross_table_hour_country_career.pivot_table(values='count', index=['country_id', 'career_id'], columns='hour', fill_value=0)

# Plot the heatmap
plt.figure(figsize=(12,6))
sns.heatmap(cross_table_hour_country_career_pivot, cmap='Blues')
plt.title('Heatmap of User Activity by Hour, Country and Career (Top 10)')
plt.xlabel('Hour')
plt.ylabel('(Country ID, Career ID)')
plt.show()

# Filter the data to include only the top 10 countries, careers and resources 用户国家、职业和资源使用情况的关系
filtered_data = dataView[dataView['country_id'].isin(top_10_countries) & dataView['career_id'].isin(top_10_careers) & dataView['resource_id'].isin(top_10_resources)]

# Create a cross table for 'country_id', 'career_id' and 'resource_id'
cross_table_country_career_resource = filtered_data.groupby(['country_id', 'career_id', 'resource_id']).size().reset_index(name='count')

# Pivot the table for the heatmap
cross_table_country_career_resource_pivot = cross_table_country_career_resource.pivot_table(values='count', index=['country_id', 'career_id'], columns='resource_id', fill_value=0)

# Plot the heatmap
plt.figure(figsize=(12,6))
sns.heatmap(cross_table_country_career_resource_pivot, cmap='Blues')
plt.title('Heatmap of Resource Usage by Country and Career (Top 10)')
plt.xlabel('Resource ID')
plt.ylabel('(Country ID, Career ID)')
plt.show()

# Get the top 10 users 访问资源最多的前10个用户
top_10_users = dataView['user_id'].value_counts().head(10)
# Plot the top 10 users
plt.figure(figsize=(12,6))
top_10_users.plot(kind='bar', color='blue')
plt.title('Top 10 Active Users')
plt.xlabel('User ID')
plt.ylabel('Count')
plt.show()

# Create a cross table for 'career_id' and 'resource_id'展示前10个职业和资源的交叉情况
cross_table_career_resource = pd.crosstab(dataView['career_id'], dataView['resource_id'])
# Select the top 10 careers and resources
cross_table_career_resource = cross_table_career_resource.loc[top_10_careers, top_10_resources]
# Plot the heatmap
plt.figure(figsize=(12,6))
sns.heatmap(cross_table_career_resource, cmap='Blues', annot=True, fmt='d')
plt.title('Heatmap of Resource Usage by Career and Resource (Top 10)')
plt.xlabel('Resource ID')
plt.ylabel('Career ID')
plt.show()