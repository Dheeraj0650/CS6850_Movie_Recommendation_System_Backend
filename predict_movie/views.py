from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from rest_framework.decorators import api_view
from rest_framework.response import Response
from scipy.sparse import csr_matrix
from fuzzywuzzy import process
# from bson.json_util import dumps
from rest_framework.views import APIView
from rest_framework.exceptions import AuthenticationFailed
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.neighbors import NearestNeighbors
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from surprise import Reader, Dataset, SVD
# , evaluate
from surprise.model_selection import cross_validate
import requests
import boto3
import csv
from io import StringIO


def weighted_rating(x, m, C):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)

def getFileFromS3(file_name):
    bucket_name = 'movierecommendationsystem'
    s3_response = s3.get_object(Bucket=bucket_name, Key=file_name)
    file_data = s3_response ["Body"].read()
    return json.loads(file_data)

def csv_string_to_dict(csv_string):
    result = []
    csv_reader = csv.DictReader(StringIO(csv_string))
    for row in csv_reader:
        result.append(row)
    return result

def getData(table_name):
    scan_params = {
        'TableName': table_name
    }
    
    response = dynamodb.scan(**scan_params)
    
    finalArr = []
    
    while True:
        items = response['Items']
        for item in items:
            finalArr.append(item)
            
        if 'LastEvaluatedKey' in response:
            scan_params['ExclusiveStartKey'] = response['LastEvaluatedKey']
            response = dynamodb.scan(**scan_params)
        else:
            break
        
    return finalArr
    
dynamodb = boto3.client("dynamodb", region_name="us-west-1")
s3 = boto3.client("s3", region_name="us-west-1")

# data = getData("final_merged_movies")
# data = getData("final_merged_movies")
# data = getData("final_merged_movies")
# data = getData("final_merged_movies")
# data = getData("final_merged_movies")
# converted_data = []
# for item in data:
#     new_item = {}
#     for key, value in item.items():
#         new_item[key] = next(iter(value.values()))
#         if isinstance(new_item[key], str) and new_item[key].replace('.', '').isdigit():
#             new_item[key] = float(new_item[key])
#         elif isinstance(new_item[key], str) and new_item[key].isdigit():
#             new_item[key] = int(new_item[key])
#     converted_data.append(new_item)

# print(converted_data)
# movies_df = pd.DataFrame(converted_data)


merged_movie_data = getFileFromS3("merged_movie_data.json")
movie_data = getFileFromS3("movie_data.json")


movies_df = pd.read_csv('final_merged_movies.csv')
movies_df['Rating_Score'] = movies_df['Rating_Score'].astype(float)
movies_df['Total_Votes'] = movies_df['Total_Votes'].astype(float)
# movies_df['Gross_USA'] = movies_df['Gross_USA'].astype(float)
# movies_df['Movie_Name'] = movies_df['Movie_Name'].astype(str)
# movies_df['Movie_Genre'] = movies_df['Movie_Genre'].astype(str)
# movies_df['Cast'] = movies_df['Cast'].astype(str)
# movies_df['overview'] = movies_df['overview'].astype(str)
# movies_df['keywords'] = movies_df['keywords'].astype(str)
movies_df['weighted_rating'] = ((movies_df['Rating_Score'] * movies_df['Total_Votes']) / (movies_df['Total_Votes'].sum()))

# Create your views here.

class SimpleRecommender:
    def __init__(self):
        pass

    def recommend_movies_by_genre(self, genre, top_n=10):
        movies_by_genre = movies_df[movies_df['Movie_Genre'].str.contains(genre)]
        movie_info = movies_by_genre.sort_values(['Gross_USA', 'weighted_rating'], ascending=False).head(top_n)
        movie_info = movie_info[['Movie_Name']].values.tolist()
        # data = []

        # # Open the JSON file for reading
        # with open('merged_movie_data.json', 'r') as file:
        #     # Load the JSON data from the file into a dictionary
        #     data = json.load(file)

        newRecommendedMovies = []
        for obj in movie_info:
            if obj[0] in merged_movie_data:
                newRecommendedMovies.append(merged_movie_data[obj[0]])

        return newRecommendedMovies


# A Sample class with init method
class CreateC_TF_IDFModel:
    # init method or constructor
    def __init__(self):
        
        movies_df['Cast'] = movies_df['Cast'].apply(lambda x: ' '.join(x.split(',')[:20]))
        movies_df['Movie_Genre'] = movies_df['Movie_Genre'].str.replace(',', ' ') # Replace commas with spaces
        self.tfidf = TfidfVectorizer(stop_words='english')

        movies_df['Movie_Genre'] = movies_df['Movie_Genre'].fillna('')

        self.tfidf_matrix = self.tfidf.fit_transform(movies_df['Movie_Name'] + ' ' + movies_df['Movie_Genre'] + ' ' + movies_df['Cast'] + ' ' + movies_df['overview'] + ' ' + movies_df['keywords'])

        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        


    def get_recommendations(self, title):
        indices = pd.Series(movies_df.index, index=movies_df['Movie_Name']).drop_duplicates()
        idx = int(indices[title])

        sim_scores = list(enumerate(self.cosine_sim[idx]))

        rating_scores = [float(score) for score in movies_df['Rating_Score']]

        weighted_scores = [(i, sim_scores[i][1] * rating_scores[i]) for i in range(len(sim_scores))]

        weighted_scores = sorted(weighted_scores, key=lambda x: x[1], reverse=True)

        movie_indices = [i[0] for i in weighted_scores[1:20]]

        movie_info = movies_df[['Movie_Name', 'weighted_rating']].iloc[movie_indices].sort_values(by='weighted_rating', ascending=False)
        movie_info = movie_info[['Movie_Name']].values.tolist()
        
        # data = {}

        # # Open the JSON file for reading
        # with open('merged_movie_data.json', 'r') as file:
        #     # Load the JSON data from the file into a dictionary
        #     data = json.load(file)

        newRecommendedMovies = []
        for movie in movie_info:
            if movie[0] in merged_movie_data:
                newRecommendedMovies.append(merged_movie_data[movie[0]])

        return json.dumps(newRecommendedMovies)
    
class CreateC_Count_VecModel:
    # init method or constructor
    def __init__(self):
        self.count_vec = CountVectorizer(stop_words='english')
        movies_df['Movie_Genre'] = movies_df['Movie_Genre'].fillna('')
        tfidf = TfidfVectorizer(stop_words='english')
        count_matrix = tfidf.fit_transform(movies_df['Movie_Name'] + ' ' + movies_df['Movie_Genre'] + ' ' + movies_df['Cast'] + ' ' + movies_df['overview'] + ' ' + movies_df['keywords'])

        self.cosine_sim = cosine_similarity(count_matrix, count_matrix)


    def get_recommendations(self, title):
        indices = pd.Series(movies_df.index, index=movies_df['Movie_Name']).drop_duplicates()
        
        idx = int(indices[title])

        sim_scores = list(enumerate(self.cosine_sim[idx]))

        rating_scores = [float(score) for score in movies_df['Rating_Score']]

        weighted_scores = [(i, sim_scores[i][1] * rating_scores[i]) for i in range(len(sim_scores))]

        weighted_scores = sorted(weighted_scores, key=lambda x: x[1], reverse=True)

        movie_indices = [i[0] for i in weighted_scores[1:20]]

        movie_info = movies_df[['Movie_Name', 'weighted_rating']].iloc[movie_indices].sort_values(by='weighted_rating', ascending=False)
        movie_info = movie_info[['Movie_Name']].values.tolist()
        
        # data = {}

        # # Open the JSON file for reading
        # with open('merged_movie_data.json', 'r') as file:
        #     # Load the JSON data from the file into a dictionary
        #     data = json.load(file)

        newRecommendedMovies = []
        for movie in movie_info:
            if movie[0] in merged_movie_data:
                newRecommendedMovies.append(merged_movie_data[movie[0]])


        return json.dumps(newRecommendedMovies)
 
class CreateKNN_TF_IDFModel:
    # init method or constructor
    def __init__(self):
        tfidf = TfidfVectorizer(stop_words='english')

        movies_df['Movie_Genre'] = movies_df['Movie_Genre'].fillna('')

        self.tfidf_matrix = tfidf.fit_transform(movies_df['Movie_Name'] + ' ' + movies_df['Movie_Genre'] + ' ' + movies_df['Cast'] + ' ' + movies_df['overview'] + ' ' + movies_df['keywords'])

        self.knn = NearestNeighbors(n_neighbors=21, algorithm='brute', metric='cosine')
        self.knn.fit(self.tfidf_matrix)

    def get_recommendations(self, title):
        indices = pd.Series(movies_df.index, index=movies_df['Movie_Name']).drop_duplicates()
        idx = int(indices[title])

        distances, indices = self.knn.kneighbors(self.tfidf_matrix[idx])

        movie_indices = indices[0][1:21]

        movie_info = movies_df[['Movie_Name', 'weighted_rating']].iloc[movie_indices].sort_values(by='weighted_rating', ascending=False)
        movie_info = movie_info[['Movie_Name']].values.tolist()
        
        # data = {}

        # # Open the JSON file for reading
        # with open('merged_movie_data.json', 'r') as file:
        #     # Load the JSON data from the file into a dictionary
        #     data = json.load(file)

        newRecommendedMovies = []
        for movie in movie_info:
            if movie[0] in merged_movie_data:
                newRecommendedMovies.append(merged_movie_data[movie[0]])


        return json.dumps(newRecommendedMovies)

class CreateKNN_Count_VecModel:
    # init method or constructor
    def __init__(self):
        count = CountVectorizer(stop_words='english')
        movies_df['Movie_Genre'] = movies_df['Movie_Genre'].fillna('')
        self.count_matrix = count.fit_transform(movies_df['Movie_Name'] + ' ' + movies_df['Movie_Genre'] + ' ' + movies_df['Cast'] + ' ' + movies_df['overview'] + ' ' + movies_df['keywords'])
        self.knn = NearestNeighbors(n_neighbors=21, algorithm='brute', metric='cosine')
        self.knn.fit(self.count_matrix)
        
    def get_recommendations(self, title):
        indices = pd.Series(movies_df.index, index=movies_df['Movie_Name']).drop_duplicates()
        idx = int(indices[title])

        distances, indices = self.knn.kneighbors(self.count_matrix[idx])

        movie_indices = indices[0][1:21]

        movie_info = movies_df[['Movie_Name', 'weighted_rating']].iloc[movie_indices].sort_values(by='weighted_rating', ascending=False)
        movie_info = movie_info[['Movie_Name']].values.tolist()
        
        # data = {}

        # # Open the JSON file for reading
        # with open('merged_movie_data.json', 'r') as file:
        #     # Load the JSON data from the file into a dictionary
        #     data = json.load(file)

        newRecommendedMovies = []
        for movie in movie_info:
            if movie[0] in merged_movie_data:
                newRecommendedMovies.append(merged_movie_data[movie[0]])

        return json.dumps(newRecommendedMovies)

class CreateCollaborativeFilteringModel:
    # init method or constructor
    def __init__(self):
        ratings = pd.read_csv('ratings_small.csv')
        credits = pd.read_csv('credits.csv')
        keywords = pd.read_csv('keywords.csv')
        movie_names = pd.read_csv('movies_metadata.csv').drop(['belongs_to_collection', 'homepage', 'imdb_id', 'poster_path', 'status', 'title', 'video'], axis=1).drop([19730, 29503, 35587])
        
        ratings = ratings.drop('timestamp', axis = 1)

        movie_names = movie_names.rename(columns={'original_title': 'title'})
        self.movie_names = movie_names[['title', 'genres']]

        movie_data = pd.concat([ratings, self.movie_names], axis=1)

        pattern = pd.DataFrame(movie_data.groupby('title')['rating'].mean())
        pattern['total number of ratings'] = pd.DataFrame(movie_data.groupby('title')['rating'].count())

        pivot = ratings.pivot(index=['userId'], columns=['movieId'], values='rating').fillna(0)

        self.mat=csr_matrix(pivot.values)

        self.knn= NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
        self.knn.fit(self.mat)
    
    def get_recommendations(self, input):
        i = process.extractOne(input, self.movie_names['title'])[2]
        d, i = self.knn.kneighbors(self.mat[i], n_neighbors=21)
        rmi = sorted(list(zip(i.squeeze().tolist(),d.squeeze().tolist())),key=lambda x: x[1])[:0:-1]
        l = []
        for val in rmi:
            l.append({'Title':self.movie_names['title'][val[0]],'Distance':val[1]})
        results = pd.DataFrame(l, index = range(1,21))
        movie_info = results[['Title']].values.tolist()

        # data = {}

        # # Open the JSON file for reading
        # with open('movie_data.json', 'r') as file:
        #     # Load the JSON data from the file into a dictionary
        #     data = json.load(file)

        newRecommendedMovies = []
        for movie in movie_info:
            if movie[0] in movie_data:
                newRecommendedMovies.append(movie_data[movie[0]])

        return json.dumps(newRecommendedMovies)
        
movieRecommendationModel1 = CreateC_TF_IDFModel()
movieRecommendationModel2 = CreateC_Count_VecModel()
movieRecommendationModel3 = CreateKNN_TF_IDFModel()
movieRecommendationModel4 = CreateKNN_Count_VecModel()
movieRecommendationModel5 = CreateCollaborativeFilteringModel()


simpleRecommenderModel = SimpleRecommender()

class IndexView(APIView):
    def post(self, request):
        return Response(values)

class PredictMovie(APIView):
    def post(self, request):
        reqData = request.data
        predictBy = reqData['predictBy'].strip()
        method = reqData['method'].strip()
        
        recommendedMovies = {}

        if method == "CONTENT BASED":
            if predictBy == "COSINE AND TF-IDF":
                recommendedMovies = movieRecommendationModel1.get_recommendations(reqData['movieName'])
            elif predictBy == "COSINE AND COUNT VEC":
                recommendedMovies = movieRecommendationModel2.get_recommendations(reqData['movieName'])
            elif predictBy == "KNN AND TF-IDF":
                recommendedMovies = movieRecommendationModel3.get_recommendations(reqData['movieName'])
            elif predictBy == "KNN AND COUNT VEC":
                recommendedMovies = movieRecommendationModel4.get_recommendations(reqData['movieName'])
        elif method == "COLLABORATIVE":
            print("innnnnnn")
            recommendedMovies = movieRecommendationModel5.get_recommendations(reqData['inputData'])


        # input = ["Toy Story", "Jumanji", "Dracula: Dead and Loving It", "Sabrina", "Forbidden Planet", "Dead Man Walking"]
        # data = {}

        # # Open the JSON file for reading
        # with open('movie_data.json', 'r') as file:
        #     # Load the JSON data from the file into a dictionary
        #     data = json.load(file)

        # newRecommendedMovies = []
        # for movie in input:
        #     if movie in data:
        #         newRecommendedMovies.append(data[movie])
        # print(newRecommendedMovies)
        # response = getData('movies_metadata')
        # print(response)
        
        return Response(recommendedMovies)
    
class MovieByGenre(APIView):
    def post(self, request):

        genre = ["Action","Sci-Fi", "Mystery", "Crime", "Drama", "Thriller", "Romance"]
        
        genreDictionary = []

        for val in genre:
            genreDictionary.append([val, simpleRecommenderModel.recommend_movies_by_genre(val)])
        
        return Response(json.dumps(genreDictionary))