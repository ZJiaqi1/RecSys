import pandas as pd
import mysql
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load the dataset
db = mysql.connect()
sql='''
select * from dl_user_resources
'''
data = pd.read_sql(sql,db)

num_rows, num_columns = data.shape
# Get the size of the dataset in memory (in bytes)
data_size_bytes = data.memory_usage(deep=True).sum()
# Convert the size from bytes to megabytes for better readability
data_size_megabytes = data_size_bytes / (1024 ** 2)
num_rows, num_columns, data_size_megabytes
print(data.head(), num_rows, num_columns, data_size_megabytes)

# Convert the 'datetime' column to datetime type
data['datetime'] = pd.to_datetime(data['datetime'])
# Group by date and count the number of user activities
daily_activity = data.groupby(data['datetime'].dt.date).size()
# Plotting the user activity over time
plt.figure(figsize=(14,6))
daily_activity.plot(linewidth=2.5)
plt.title('User Activity Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Activities')
plt.grid(True)
plt.tight_layout()
plt.show()

# Group by country and count the number of user activities
country_activity = data.groupby('country_id').size().sort_values(ascending=False).head(10)

# Plotting the user activity by country
plt.figure(figsize=(14,6))
country_activity.plot(kind='bar', color='salmon')
plt.title('User Activity by Country')
plt.xlabel('Country ID')
plt.ylabel('Number of Activities')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Convert the 'date_added' column to datetime type
data['date_added'] = pd.to_datetime(data['date_added'])
# Group by month and count the number of resources added
monthly_added_resources = data.groupby(data['date_added'].dt.to_period("M")).size()
# Plotting the number of resources added over time
plt.figure(figsize=(14,6))
monthly_added_resources.plot(kind='bar', color='skyblue')
plt.title('Distribution of Resources Added Over Time')
plt.xlabel('Month')
plt.ylabel('Number of Resources Added')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Group by career and count the number of user activities
career_activity = data.groupby('career_id').size().sort_values(ascending=False).head(10)
# Plotting the user activity by career
plt.figure(figsize=(14,6))
career_activity.plot(kind='bar', color='lightgreen')
plt.title('User Activity by Career')
plt.xlabel('Career ID')
plt.ylabel('Number of Activities')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Group by resource and count the number of accesses
popular_resources = data.groupby('title').size().sort_values(ascending=False).head(10)
# Plotting the most popular resources
plt.figure(figsize=(14,6))
popular_resources.plot(kind='barh', color='orchid')
plt.title('Top 10 Most Popular Resources')
plt.xlabel('Number of Accesses')
plt.ylabel('Resource Title')
plt.grid(axis='x')
plt.tight_layout()
plt.show()

# Concatenate all meta descriptions
all_descriptions = ' '.join(data['mata_description'].dropna())
# Generate a word cloud
wordcloud = WordCloud(width = 800, height = 400, background_color ='white').generate(all_descriptions)
# Plotting the word cloud
plt.figure(figsize=(14,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Resource Description Word Cloud')
plt.tight_layout()
plt.show()

# Basic descriptive statistics for the dataset
desc_stats = data.describe(include='all').transpose()
desc_stats[['count', 'unique', 'top', 'freq']]

# 用户的活跃度分析
# Analyzing user activity
user_activity = data['user_id'].value_counts()
# Descriptive statistics for user activity
user_activity_desc = user_activity.describe()
# Displaying the top 10 most active users and the activity statistics
top_10_users = user_activity.head(10)
print(user_activity_desc, top_10_users)

# 资源的受欢迎程度
# Analyzing resource popularity
resource_popularity = data['resource_id'].value_counts()
# Descriptive statistics for resource popularity
resource_popularity_desc = resource_popularity.describe()
# Displaying the top 10 most popular resources and the popularity statistics
top_10_resources = resource_popularity.head(10)
print(resource_popularity_desc, top_10_resources)

#按国家和职业的活跃度分析
# Analyzing activity by country
country_activity = data['country_id'].value_counts()
# Analyzing activity by career
career_activity = data['career_id'].value_counts()
# Displaying the top 5 most active countries and careers
top_5_countries = country_activity.head(5)
top_5_careers = career_activity.head(5)
print(top_5_countries, top_5_careers)