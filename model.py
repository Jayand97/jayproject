import pandas as pd
import numpy as np
import pickle
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

columns=['userId', 'productId', 'ratings','timestamp']
electronics_df=pd.read_csv('ratings_Electronics.csv',names=columns)

electronics_df1 = electronics_df.iloc[:50000,0:]

most_rated = electronics_df1.groupby('userId').size().sort_values(ascending=False)[:10]

counts = electronics_df1.userId.value_counts()
electronics_df1_final = electronics_df1[electronics_df1.userId.isin(counts[counts>=15].index)]

final_ratings_matrix = electronics_df1_final.pivot(index = 'userId', columns ='productId', values = 'ratings').fillna(0)

train_data = electronics_df1_final
train_data_grouped = train_data.groupby('productId').agg({'userId': 'count'}).reset_index()

train_data_grouped.rename(columns = {'userId': 'score'},inplace=True)

train_data_grouped.head()

#Sort the products on recommendation score 
train_data_sort = train_data_grouped.sort_values(['score', 'productId'], ascending = [0,1]) 
      
#Generate a recommendation rank based upon score 
train_data_sort['rank'] = train_data_sort['score'].rank(ascending=0, method='first') 
          
#Get the top 5 recommendations 
popularity_recommendations = train_data_sort.head()

pickle.dump(popularity_recommendations, open('model.pkl', 'wb'))
pickle.dump(electronics_df1_final, open('eldf_final.pkl', 'wb'))
pickle.dump(final_ratings_matrix, open('final_ratings_matrix.pkl', 'wb'))