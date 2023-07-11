import matplotlib.pyplot as plt
import mysql
import pandas as pd

# Load the data
db = mysql.connect()
sql='''
select * from user_hashed
'''
data = pd.read_sql(sql,db)

# Display the first few rows of the dataframe
data.head()

# Create a figure with two subplots
fig, ax = plt.subplots(2, 1, figsize=(10, 12))

# Plot histogram for country_id
data['country_id'].plot(kind='hist', bins=20, ax=ax[0], color='skyblue', edgecolor='black')
ax[0].set_title('Histogram of country_id')
ax[0].set_xlabel('Country ID')
ax[0].set_ylabel('Frequency')

# Plot histogram for career_id
data['career_id'].plot(kind='hist', bins=20, ax=ax[1], color='skyblue', edgecolor='black')
ax[1].set_title('Histogram of career_id')
ax[1].set_xlabel('Career ID')
ax[1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()
