from django.shortcuts import render
from django.http import HttpResponse
from django.contrib.auth import authenticate
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from string import punctuation
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
#from bokeh.io import show, output_file
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from bokeh.embed import components
import numpy as np
# Create your views here.

badfood=pickle.load(open('mymodel/pkl/BadFood.pickle','rb'))
cost=pickle.load(open('mymodel/pkl/Cost.pickle','rb'))
filthy=pickle.load(open('mymodel/pkl/Filthy.pickle','rb'))
missing_food=pickle.load(open('mymodel/pkl/MissingFood.pickle','rb'))
order_problem=pickle.load(open('mymodel/pkl/OrderProblem.pickle','rb'))
rude_service=pickle.load(open('mymodel/pkl/RudeService.pickle','rb'))
scary_mcd = pickle.load(open('mymodel/pkl/ScaryMcDs.pickle', 'rb'))
slow_service=pickle.load(open('mymodel/pkl/SlowService.pickle','rb'))

with open('mymodel/pkl/vectorizer.pkl', 'rb') as file: 
    vectorizer = pickle.load(file)

def index(request):
    return render(request,'mymodel/index.html')



def remove_non_ascii_1(text):

    return ''.join(i for i in text if ord(i)<128)


def clean_text(input_str):
    lemmatizer= WordNetLemmatizer()
    input_str=input_str.lower()
    remove_num=re.sub(r'\d+','',input_str)
    remove_punc=remove_num.translate(str.maketrans("","",punctuation))
    remove_white=remove_punc.strip()
    stop_words=set(stopwords.words('english'))
    tokens=word_tokenize(remove_white)
    result=[i for i in tokens if  not i in stop_words]
    lemmatized_words=[lemmatizer.lemmatize(word) for word in result]


    review=' '.join(lemmatized_words)
    return review


def predict(request):
    # Grabbing data from user

    if request.method=="POST":
        review=request.POST.get('review')
        city=request.POST.get('city')

        input_data=[{'review':review,'city':city}]
        dataset=pd.DataFrame(input_data)
        dataset = dataset.replace(r'\r',' ', regex=True)
        dataset['review']=dataset['review'].apply(lambda x:remove_non_ascii_1(x))
        dataset['review']=dataset['review'].apply(lambda x:clean_text(x))

    # Bag of words
        features_data = pd.DataFrame(vectorizer.transform(dataset.review).toarray())
        features_data.columns=vectorizer.get_feature_names()
        features_data.insert(0,'city_x',dataset['city'])

    # Label Encoding the city column
        labelencoder=LabelEncoder()
        features_data['city_x']=labelencoder.fit_transform(features_data.city_x)
        features_data['city_x']=features_data['city_x'].astype('category')
        
        regressor_1=badfood.predict(features_data)
        regressor_2 = cost.predict(features_data)
        regressor_3 = filthy.predict(features_data)
        regressor_4 = missing_food.predict(features_data)
        regressor_5 = order_problem.predict(features_data)
        regressor_6 = rude_service.predict(features_data)
        regressor_7 = scary_mcd.predict(features_data)
        regressor_8 = slow_service.predict(features_data)
        confidence=[regressor_1,regressor_2,regressor_3,regressor_4,regressor_5,
        regressor_6,regressor_7,regressor_8]
        policies=['BadFood','Cost','Filthy','MissingFood','OrderProblem',
        'RudeService','ScaryMcDs','SlowService']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
          '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
          '#bcbd22']
          

        source = ColumnDataSource(data=dict(policies=policies, confidence=confidence))

        p = figure(x_range=policies, plot_height=500,
                toolbar_location=None, title="Policies Violated Analysis",plot_width=800)
        p.vbar(x='policies', top='confidence', width=0.9, source=source, legend="policies",
            line_color='black', fill_color=factor_cmap('policies', palette=colors, factors=policies))

        p.xgrid.grid_line_color = None
        p.y_range.start = 0.1
        p.y_range.end = 0.9
        p.legend.orientation = "horizontal"
        p.legend.location = "top_center"
        p.xaxis.axis_label = 'Policies Violated'
        p.yaxis.axis_label='Policies Violated:Confidence'
        script,div=components(p)

    return render(request,'mymodel/result.html',{'script':script,'div':div})


def predict_upload(request):
    if request.method=='POST':
        file = request.FILES['myfile']
        dataset = pd.read_csv(file, encoding='latin-1')
        
        dataset = dataset.replace(r'\r', ' ', regex=True)
        
        dataset['review']=dataset['review'].apply(lambda x:remove_non_ascii_1(x))
        dataset['review']=dataset['review'].apply(lambda x:clean_text(x))
        
            # Bag of words
        features_data = pd.DataFrame(vectorizer.transform(dataset.review).toarray())
        features_data.columns=vectorizer.get_feature_names()
        features_data.insert(0,'city_x',dataset['city'])
        
            # Label Encoding the city column
        labelencoder=LabelEncoder()
        features_data['city_x']=labelencoder.fit_transform(features_data.city_x)
        features_data['city_x']=features_data['city_x'].astype('category')
        
                
        regressor_1=badfood.predict(features_data)
        regressor_1=np.sum(regressor_1)
        regressor_2 = cost.predict(features_data)
        regressor_2=np.sum(regressor_2)
        regressor_3 = filthy.predict(features_data)
        regressor_3 = np.sum(regressor_3)
        regressor_4 = missing_food.predict(features_data)
        regressor_4=np.sum(regressor_4)
        regressor_5 = order_problem.predict(features_data)
        regressor_5=np.sum(regressor_5)
        regressor_6 = rude_service.predict(features_data)
        regressor_6=np.sum(regressor_6)
        regressor_7 = scary_mcd.predict(features_data)
        regressor_7=np.sum(regressor_7)
        regressor_8 = slow_service.predict(features_data)
        regressor_8=np.sum(regressor_8)
        

        confidence=[regressor_1,regressor_2,regressor_3,regressor_4,regressor_5,
        regressor_6,regressor_7,regressor_8]
        
        policies=['BadFood','Cost','Filthy','MissingFood','OrderProblem',
                'RudeService','ScaryMcDs','SlowService']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                '#bcbd22']
                

        source = ColumnDataSource(data=dict(policies=policies, confidence=confidence))

        p = figure(x_range=policies, plot_height=500,
                        toolbar_location=None, title="Policies Violated Analysis",plot_width=800)
        p.vbar(x='policies', top='confidence', width=0.9, source=source, legend="policies",
                    line_color='black', fill_color=factor_cmap('policies', palette=colors, factors=policies))

        p.xgrid.grid_line_color = None
        p.y_range.start = 0.1
        p.y_range.end =10.0 
        p.legend.orientation = "horizontal"
        p.legend.location = "top_center"
        p.xaxis.axis_label = 'Policies Violated'
        p.yaxis.axis_label='Policies Violated:Confidence'
        script,div=components(p)
        
    return render(request,'mymodel/result.html',{'script':script,'div':div})
    





        
