import numpy as np
import pandas as pd

movie = pd.read_csv('dataset.csv')

movie = movie[['id', 'title', 'overview', 'genre']]

movie['tags'] = movie['overview'] + movie['genre']
new_data = movie.drop(columns=['overview','genre'])

from sklearn.feature_extraction.text import CountVectorizer 

cv = CountVectorizer(max_features=10000, stop_words = 'english')

vector = cv.fit_transform(new_data['tags'].values.astype('U')).toarray()

from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vector)  

def recommand(movie):
    index = new_data[new_data['title'] == movie].index[0]
    distance = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda vector:vector[1] )
    for i in distance[0:5]:
        print(new_data.iloc[i[0]].title)

a = input("Enter the film's name:")
recommand(a)