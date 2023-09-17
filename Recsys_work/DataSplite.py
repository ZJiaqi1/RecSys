import mysql
import pandas as pd

# Load the dataset
db = mysql.connect()
sql='''
select * from dl_user_resources
'''
data = pd.read_sql(sql,db)
# Convert the 'datetime' column to datetime data type for accurate sorting
data['datetime'] = pd.to_datetime(data['datetime'])
# Sort the DataFrame first by 'user_id' and then by 'datetime'
df_sorted = data.sort_values(by=['user_id', 'datetime'])

# Group the data by 'user_id' and find the index of the row with the maximum 'datetime' for each user
test_indices = df_sorted.groupby('user_id')['datetime'].idxmax()
# Extract the test set using these indices
test_set = df_sorted.loc[test_indices]
# Extract the training set by dropping these indices from the original DataFrame
train_set = df_sorted.drop(test_indices)
# Display the first few rows of the training and test sets
train_set.head(), test_set.head()