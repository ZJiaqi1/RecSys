import pandas as pd
import mysql
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Load the dataset
db = mysql.connect()
sql='''
select * from dl_user_resources limit 0,3000
'''
data = pd.read_sql(sql,db)

# Re-define the TF-IDF vectorizer and the scaler
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
scaler = MinMaxScaler()

content = data['content'] = data['title'].fillna('') + ' ' + data['meta_title'].fillna('') + ' ' + data['mata_description'].fillna('')

# Compute TF-IDF matrix
tfidf_matrix = tfidf_vectorizer.fit_transform(content)

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Construct a reverse map of indices and resource titles
indices = pd.Series(data.index, index=data['resource_id']).drop_duplicates()

# Normalize the cosine similarity scores to be between 0 and 1
cosine_sim = scaler.fit_transform(cosine_sim)

cosine_sim.shape, indices.head()

def get_recommendations(resource_id, cosine_sim=cosine_sim, data=data, indices=indices):
    # Get the index of the resource that matches the resource_id
    idx = indices[resource_id]

    # Get the pairwsie similarity scores of all resources with that resource
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the resources based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar resources
    sim_scores = sim_scores[1:11]

    # Get the resource indices
    resource_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar resources
    return data['title'].iloc[resource_indices]

# Test the function with a random resource_id
#get_recommendations('test')
