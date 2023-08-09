import pandas as pd
import mysql
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
db = mysql.connect()
sql='''
select * from user_hashed
'''
data = pd.read_sql(sql,db)
# Display the first few rows of the data
print(data.head())

# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# Plotting the distribution of users by country
plt.figure(figsize=(12, 6))
country_counts = data['country_id'].value_counts().head(10)
sns.barplot(x=country_counts.index, y=country_counts.values, palette="viridis")
plt.title('Top 10 Countries by User Count')
plt.xlabel('Country ID')
plt.ylabel('Number of Users')
plt.show()

# Plotting the distribution of users by career
plt.figure(figsize=(12, 6))
career_counts = data['career_id'].value_counts().head(10)
sns.barplot(x=career_counts.index, y=career_counts.values, palette="viridis")
plt.title('Top 10 Careers by User Count')
plt.xlabel('Career ID')
plt.ylabel('Number of Users')
plt.show()

# Selecting the top 5 countries
top_countries = country_counts.head(5).index
# Filtering data for these countries
filtered_data = data[data['country_id'].isin(top_countries)]
# Creating a cross table to count users by country and career
ct = pd.crosstab(filtered_data['country_id'], filtered_data['career_id'])
# Plotting the most popular career for each of the top 5 countries
plt.figure(figsize=(14, 7))
for country in top_countries:
    most_popular_career = ct.loc[country].idxmax()
    max_users = ct.loc[country].max()
    sns.barplot(x=[most_popular_career], y=[max_users], label=f'Country {country}', alpha=0.7)
plt.title('Most Popular Career in Top 5 Countries')
plt.xlabel('Career ID')
plt.ylabel('Number of Users')
plt.legend()
plt.show()

# Plotting the career distribution for the top 5 countries using stacked bar chart
ct_top_careers = ct[ct.sum(axis=1) > 0].head(10)
ct_top_careers.plot(kind='bar', stacked=True, figsize=(14, 7), colormap="viridis")
plt.title('Career Distribution in Top 5 Countries')
plt.xlabel('Country ID')
plt.ylabel('Number of Users')
plt.show()

# Boxplot to show the career distribution for each of the top 5 countries
plt.figure(figsize=(14, 7))
sns.boxplot(x='country_id', y='career_id', data=filtered_data, palette="viridis")
plt.title('Career Distribution by Country (Boxplot)')
plt.xlabel('Country ID')
plt.ylabel('Career ID')
plt.show()

# Heatmap to show the relationship between the top 5 countries and top 10 careers
plt.figure(figsize=(14, 7))
top_careers = career_counts.head(10).index
heatmap_data = pd.crosstab(filtered_data['country_id'], filtered_data['career_id'])[top_careers]
sns.heatmap(heatmap_data, cmap="viridis", annot=True, fmt="d")
plt.title('User Count by Country and Career (Heatmap)')
plt.xlabel('Career ID')
plt.ylabel('Country ID')
plt.show()

# Pie chart to show the distribution of users by country for the top 5 countries
plt.figure(figsize=(10, 6))
labels = top_countries
sizes = country_counts.head(5).values
colors = sns.color_palette("viridis", len(labels))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title('User Distribution by Country (Pie Chart)')
plt.show()

# Histogram to show the distribution of careers
plt.figure(figsize=(12, 6))
sns.histplot(data['career_id'], bins=30, kde=True, color="blue")
plt.title('Distribution of Career IDs (Histogram)')
plt.xlabel('Career ID')
plt.ylabel('Frequency')
plt.show()

# Violin plot to show the distribution of careers for each of the top 5 countries
plt.figure(figsize=(14, 7))
sns.violinplot(x='country_id', y='career_id', data=filtered_data, palette="viridis")
plt.title('Career Distribution by Country (Violin Plot)')
plt.xlabel('Country ID')
plt.ylabel('Career ID')
plt.show()
