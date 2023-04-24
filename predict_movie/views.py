from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from rest_framework.decorators import api_view
from rest_framework.response import Response
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

dynamodb = boto3.client("dynamodb", region_name="us-west-1")
s3 = boto3.client("s3")

# Create your views here.
values = dict()
values['a'] = 10
values['b'] = 20

# # @require_POST
# def index(request):
#     print("hello")
#     print(request)
#     print("bye")
    
#     if request.method == 'POST':
#         # Handle the POST request
#         data = request.POST.get('data');
#         print(data)
#         # Do something with the data
#         response_data = {'success': True}
#         return JsonResponse(response_data)
#     else:
#         return HttpResponse(values.items())
    

# def handlePost(request):
#     print("hey")
#     return HttpResponse("this is post request")

# Register User API

# A Sample class with init method
class CreateModel:
    # init method or constructor
    def __init__(self):
        self.md = pd.read_csv('./movies_metadata.csv')
        self.md['genres'] = self.md['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
        self.vote_counts = self.md[self.md['vote_count'].notnull()]['vote_count'].astype('int')
        self.vote_averages = self.md[self.md['vote_average'].notnull()]['vote_average'].astype('int')
        self.C = self.vote_averages.mean()
        self.m = self.vote_counts.quantile(0.95)
        self.md['year'] = pd.to_datetime(self.md['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
        self.qualified = self.md[(self.md['vote_count'] >= self.m) & (self.md['vote_count'].notnull()) & (self.md['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']]
        self.qualified['vote_count'] = self.qualified['vote_count'].astype('int')
        self.qualified['vote_average'] = self.qualified['vote_average'].astype('int')

        def weighted_rating(x):
            self.v = x['vote_count']
            self.R = x['vote_average']
            return (self.v/(self.v+self.m) * self.R) + (self.m/(self.m+self.v) * self.C)
        
        self.qualified['wr'] = self.qualified.apply(weighted_rating, axis=1)
        
        self.qualified = self.qualified.sort_values('wr', ascending=False).head(250)

        self.s = self.md.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
        self.s.name = 'genre'
        self.gen_md = self.md.drop('genres', axis=1).join(self.s)

        def build_chart(genre, percentile=0.85):
            self.df = self.gen_md[self.gen_md['genre'] == genre]
            self.vote_counts = self.df[self.df['vote_count'].notnull()]['vote_count'].astype('int')
            vote_averages = self.df[self.df['vote_average'].notnull()]['vote_average'].astype('int')
            self.C = self.vote_averages.mean()
            self.m = self.vote_counts.quantile(percentile)
            
            self.qualified = self.df[(self.df['vote_count'] >= self.m) & (self.df['vote_count'].notnull()) & (self.df['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity']]
            self.qualified['vote_count'] = self.qualified['vote_count'].astype('int')
            self.qualified['vote_average'] = self.qualified['vote_average'].astype('int')
            
            self.qualified['wr'] = self.qualified.apply(lambda x: (x['vote_count']/(x['vote_count']+self.m) * x['vote_average']) + (self.m/(self.m+x['vote_count']) * self.C), axis=1)
            self.qualified = self.qualified.sort_values('wr', ascending=False).head(250)
            
            return self.qualified
        
        self.links_small = pd.read_csv('links_small.csv')
        self.links_small = self.links_small[self.links_small['tmdbId'].notnull()]['tmdbId'].astype('int')

        self.md = self.md.drop([19730, 29503, 35587])

        self.md['id'] = self.md['id'].astype('int')
        self.smd = self.md[self.md['id'].isin(self.links_small)]

        self.smd['tagline'] = self.smd['tagline'].fillna('')
        self.smd['description'] = self.smd['overview'] + self.smd['tagline']
        self.smd['description'] = self.smd['description'].fillna('')

        self.tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
        self.tfidf_matrix = self.tf.fit_transform(self.smd['description'])

        self.cosine_sim = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)

        self.smd = self.smd.reset_index()
        self.titles = self.smd['title']
        self.indices = pd.Series(self.smd.index, index=self.smd['title'])

    def get_recommendations(self, title):
        idx = self.indices[title]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:31]
        movie_indices = [i[0] for i in sim_scores]
        movie_info = self.titles.iloc[movie_indices]
        recommendations = []
        for index, row in movie_info.to_frame().iterrows():
            movie_dict = {}
            movie_dict['title'] = row['title']
            recommendations.append(movie_dict)
        return json.dumps(recommendations)
        
movieRecommendationModel = CreateModel()

class IndexView(APIView):
    def post(self, request):
        return Response(values)


def weighted_rating(x, m, C):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)

def getFileFromS3(file_name):
    bucket_name = 'movierecommendationsystem'
    s3_response = s3.get_object(Bucket=bucket_name, Key=file_name)
    file_data = s3_response ["Body"].read().decode('utf')
    return csv_string_to_dict(file_data)

def csv_string_to_dict(csv_string):
    result = []
    csv_reader = csv.DictReader(StringIO(csv_string))
    for row in csv_reader:
        result.append(row)
    return result

def getData(table_name):
    # set up the initial parameters for the scan operation
    scan_params = {
        'TableName': table_name
    }
    
    # initiate the scan operation
    response = dynamodb.scan(**scan_params)
    
    finalArr = []
    
    while True:
        items = response['Items']
        for item in items:
            finalArr.append(item)
            
        # check if there are more items to retrieve
        if 'LastEvaluatedKey' in response:
            # set the exclusive start key to the last evaluated key
            scan_params['ExclusiveStartKey'] = response['LastEvaluatedKey']
            # initiate another scan operation
            response = dynamodb.scan(**scan_params)
        else:
            # all items have been retrieved
            break
        
    # for item in finalArr:
    #     for key, value in item.items():
    #         item[key] = value.get(list(value.keys())[0])
    
    return finalArr

class PredictMovie(APIView):
    def post(self, request):
        values['a'] = 40
        values['b'] = 50
        reqData = request.data

        recommendedMovies = movieRecommendationModel.get_recommendations(reqData['movieName'])
        response = getData('credits')
        print(response)
        print(recommendedMovies)
        return Response(recommendedMovies)