
# Importing the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import pickle
random.seed(123)
# Importing the dataset
dataset=pd.read_csv('McDonalds-Yelp-Sentiment-DFE.csv',encoding='latin-1',na_values='na')

# Exploratory Data Analysis

dataset['_last_judgment_at']=pd.to_datetime(dataset['_last_judgment_at'])


print(dataset.isnull().sum()/len(dataset))

dataset=dataset.drop('policies_violated_gold',axis=1)
print(dataset.dtypes)
print(dataset['city'].value_counts())

print(dataset.head(5))

# Visualising the city column
from bokeh.io import show, output_file
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.transform import factor_cmap

labels=[]
for city in dataset['city'].unique():
    labels.append(city)

counts=[]
for count in dataset['city'].value_counts().unique():
    counts.append(count)

labels=labels[0:9]

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
          '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
          '#bcbd22']
          
#output_file("bar_colormapped.html")
source = ColumnDataSource(data=dict(labels=labels, counts=counts))

p = figure(x_range=labels, plot_height=350, toolbar_location=None, title="City Counts")
p.vbar(x='labels', top='counts', width=0.9, source=source, legend="labels",
       line_color='white', fill_color=factor_cmap('labels', palette=colors,factors=labels))

p.xgrid.grid_line_color = None
p.y_range.start = 0
p.y_range.end = 500
p.legend.orientation = "horizontal"
p.legend.location = "top_center"
#show(p)

dataset['city'].fillna('Atlanta',inplace=True)

print(dataset['policies_violated:confidence'].head(5))
print(dataset['policies_violated'].value_counts())

#dataset['policies_violated']=dataset['policies_violated'].replace('na',np.NaN)
# Cleaning the policies_violated column



print(dataset['policies_violated'].head(10))

dataset=dataset.dropna(subset=['policies_violated'])

dataset=dataset.reset_index(drop=True)



#dataset_new=dataset.pivot(index='_unit_id',columns='policies_violated_new',values='policies_violated:confidence')
#confidence_new_list=[]
#confidence_new_list=dataset['policies_violated:confidence'].apply(lambda x: x.split(' '))

dataset = dataset.replace(r'\r',' ', regex=True) 

#dataset['policies_violated_new']=dataset['policies_violated_new'].groupby('')
#dataset['confidence_new']=nums



#dataset_new=dataset_new.astype('float')
#dataset.drop(['index','_unit_id','_golden','_unit_state','_trusted_judgments','_last_judgment_at'],axis=1,inplace=True)
#dataset_new.drop('na',axis=1,inplace=True)
#dataset_new=dataset_new.drop('na',axis=1)

#print(dataset_new.isnull().sum()/len(dataset_new))
#dataset['confidence_new'].astype('float',inplace=True)
#dataset_new=pd.DataFrame(dataset_new)


'''
dataset=pd.merge(dataset,dataset_new,on='index')

dataset.drop(['_unit_id','_golden','_trusted_judgments','_last_judgment_at'],axis=1,inplace=True)

dataset.drop(['_unit_state','policies_violated','policies_violated:confidence','policies_violated_new','confidence_new'],axis=1,inplace=True)
dataset=dataset.reset_index()

dataset.drop('level_0',axis=1,inplace=True)

# Cleaning the Reviews column of the dataset
reviews=dataset['review']
'''



dataset.columns

dataset.policies_violated.head()
dataset['policies_violated:confidence'].head()

dataset['policies_violated_list']=dataset['policies_violated'].apply(lambda x: x.split())

policy=dataset.loc[0,'policies_violated_list']
confidence=dataset.loc[0,'policies_violated:confidence']
confidence=confidence.split()

dataset['confidence_list']=dataset['policies_violated:confidence'].apply(lambda x: x.split())


#dict(zip(dataset['policies_violated_list'].tolist(),dataset['confidence_list'].tolist()))

for i in range(0,len(dataset)):
    
    policy=dataset.loc[i,'policies_violated_list']
    confidence=dataset.loc[i,'confidence_list']
    dictionary=dict(zip(policy,confidence))
    for key in dictionary.keys():
        dataset.loc[i,key]=dictionary[key]
        
    #dataset.loc[i,'dictionary']=str(dictionary)
    
dataset.fillna(0,inplace=True)    
    
    
dataset.drop([ '_golden', '_unit_state', '_trusted_judgments',
       '_last_judgment_at', 'policies_violated',
       'policies_violated:confidence',
       'policies_violated_list', 'confidence_list','na'],axis=1,inplace=True)    
    
    
# Cleaning the review column    
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from string import punctuation
from nltk.corpus import stopwords
#text_col=dataset['review_x']

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
    
    #for word in result:
     #   return lemmatizer.lemmatize(word)
    cleaned_text=' '.join(lemmatized_words)
    
    return cleaned_text

dataset['cleaned_text']=dataset['review'].apply(lambda x: remove_non_ascii_1(x))

dataset['cleaned_text']=dataset['cleaned_text'].apply(lambda x: clean_text(x))
 
#text_col=text_col.apply(clean_text)
#text=text_col
#text_colnames=text_col
#text_colnames=text_colnames.unique()

# Bag of Words
from sklearn.feature_extraction.text import CountVectorizer
vectorizer=CountVectorizer()



X_train_counts = vectorizer.fit_transform(dataset.cleaned_text)
#X_train_counts = vectorizer.fit_transform(dataset.review)
#dictionary=vectorizer.vocabulary_
features_data=pd.DataFrame(X_train_counts.toarray())

features_data.columns=vectorizer.get_feature_names()

#features_data.insert(0,'city_x',dataset['city'])

features_data['_unit_id']=dataset['_unit_id']
dataset=pd.merge(dataset,features_data,on='_unit_id')

print(dataset.dtypes)

print(dataset.isnull().sum()/len(dataset))


# Label encoding the city column
#print(dataset['city_x'].unique())
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()

dataset['city_x']=labelencoder.fit_transform(dataset['city_x'])

dataset.drop(['_unit_id','review_x','cleaned_text'],axis=1,inplace=True)

# Changing the datatype of object column

dataset[dataset.select_dtypes(['object']).columns]=dataset.select_dtypes(['object']).apply(lambda x: x.astype('float'))

# Changing datatype of city_x column
dataset['city_x']=dataset['city_x'].astype('category')


# Splitting the dataset into dependent and independent variables
X=dataset.drop(['RudeService' ,'OrderProblem', 'Filthy' ,'SlowService',            
'BadFood' ,               
'ScaryMcDs', 'MissingFood',
'Cost'],axis=1)

y=dataset.loc[:,['RudeService' ,'OrderProblem', 'Filthy' ,'SlowService',            
'BadFood' ,               
'ScaryMcDs', 'MissingFood',
'Cost']]

# Splitting the dataset into the Train and Test
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=123)


#X_train.to_csv('Train.csv')

#X_train_data=pd.read_csv('Train.csv')
#X_train_data.drop('Unnamed: 0',axis=1,inplace=True)
# Training on the train dataset

#from sklearn.multioutput import MultiOutputRegressor
#from sklearn.linear_model import LinearRegression
#from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
#from xgboost import XGBRegressor
#regr=RandomForestRegressor(n_estimators=100,max_depth=1,min_samples_split=7,min_samples_leaf=6)
regr=RandomForestRegressor(max_depth=2)
#regr_multi=MultiOutputRegressor(RandomForestRegressor())
#regr_multi.fit(X_train,y_train)
#print(regr.get_params)

regr.fit(X_train,y_train['BadFood'])

# Predicting the train set results
y_train_pred=regr.predict(X_train)
#y_train_pred=pd.DataFrame(y_train_pred)
#y_train_pred.index=y_train.index
#y_train_pred.columns=y_train.columns
print(np.sqrt(mean_squared_error(y_train_pred, y_train['BadFood'])))




# Predicting the test set results
y_pred=regr.predict(X_test)
#y_pred=pd.DataFrame(y_pred)
#y_pred.index=y_test.index
#y_pred.columns=y_test.columns

# RMSE and MAE
from sklearn.metrics import mean_squared_error
#from sklearn.metrics import mean_absolute_error
#from sklearn.metrics import max_error
print(np.sqrt(mean_squared_error(y_test['BadFood'], y_pred)))

# Cost-0.03
# RudeService-0.01
# ScaryMcDs-0.01
# Filthy-0.001
# OrderProblem-0.01
# MissingFood-0.03
# SlowService-0.001
# BadFood=0.01



#print(mean_absolute_error(y_test,y_pred,multioutput='raw_values'))
#y_pred_head=y_pred.head(10)

#y_test_head=y_test.head(10)

# Visualising the y_pred and y_test
#y_pred_plot=y_pred_head.plot.bar(stacked=True)
#plt.savefig('predictions.png')

#y_test_plot=y_test_head.plot.bar(stacked=True)
#plt.savefig('test_predictions.png')

# K-Fold cross validation
#from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer
mse=make_scorer(mean_squared_error)
#mae=make_scorer(mean_absolute_error)
#rmse=make_scorer(np.sqrt(mean_absolute_error))
#kf=KFold(n_splits=10,random_state=123)
#scores=cross_val_score(regr,X_train,y_train,cv=10,n_jobs=10,scoring=mse)
#print(scores)

# Grid Search for hyperparameter tuning for RandomForest
from sklearn.model_selection import GridSearchCV
#print(regr.get_params)
parameters={'n_estimators':[100,200,300,400,500,600]}

grid_param=GridSearchCV(estimator=regr,param_grid=parameters,cv=5,scoring=mse)

grid_param=grid_param.fit(X_train,y_train['Cost'])

print(grid_param.best_params_)

#print(regr.get_params)

# Saving the model in pickle file
pickle.dump(regr,open('Cost.pickle','wb'))


# Saving the vectorizer in pickle file

with open('vectorizer.pkl', 'wb') as file:
     pickle.dump(vectorizer, file)







