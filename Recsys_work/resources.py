import pandas as pd
import mysql
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# Load the dataset
db = mysql.connect()
sql='''
select * from resources_hashed
'''
data = pd.read_sql(sql,db)
# Display the first few rows of the data
# Calculate the number of rows and columns in the dataset
num_rows, num_columns = data.shape
# Get the size of the dataset in memory (in bytes)
data_size_bytes = data.memory_usage(deep=True).sum()
# Convert the size from bytes to megabytes for better readability
data_size_megabytes = data_size_bytes / (1024 ** 2)
num_rows, num_columns, data_size_megabytes
print(data.head(), num_rows, num_columns, data_size_megabytes)

data_sorted_date = data.sort_values('date_added')
# Optimizing the plot by resampling the data on a monthly basis
data_sorted_date.set_index('date_added', inplace=True)
monthly_data = data_sorted_date.resample('M').size().cumsum()
# Plot
plt.figure(figsize=(14, 7))
monthly_data.plot()
plt.title('Cumulative Resources Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Resources')
plt.tight_layout()
plt.show()


# Filter out invalid date values
valid_dates = data_sorted_date['date_updated'].apply(lambda x: str(x).startswith("0000-00-00") == False)
data_cleaned_dates = data_sorted_date[valid_dates]
# Resetting the index before resampling
data_cleaned_dates_reset = data_cleaned_dates.reset_index()
# Filtering out invalid dates and converting the 'date_updated' column to datetime format
data_cleaned_dates_reset['date_updated'] = pd.to_datetime(data_cleaned_dates_reset['date_updated'], errors='coerce')
# Dropping NaT values from 'date_updated' after conversion
data_cleaned_dates_reset = data_cleaned_dates_reset.dropna(subset=['date_updated'])
# Resample the cleaned data on a yearly basis for updates
yearly_updates = data_cleaned_dates_reset.resample('Y', on='date_updated').size()
# Plot
plt.figure(figsize=(14, 7))
yearly_updates.plot(kind='bar')
plt.title('Number of Resources Updated Each Year')
plt.xlabel('Year')
plt.ylabel('Number of Updates')
plt.tight_layout()
plt.show()

# Calculate the length of each title
data['title_length'] = data['title'].str.len()

# Plot
plt.figure(figsize=(14, 7))
sns.histplot(data['title_length'], bins=50, kde=True)
plt.title('Distribution of Resource Title Lengths')
plt.xlabel('Title Length')
plt.ylabel('Number of Resources')
plt.tight_layout()
plt.show()

# Resample the data on a monthly basis for additions
monthly_additions = data_sorted_date.resample('M').size()
# Plot
plt.figure(figsize=(14, 7))
monthly_additions.plot()
plt.title('Number of Resources Added Each Month')
plt.xlabel('Month')
plt.ylabel('Number of Resources Added')
plt.tight_layout()
plt.show()