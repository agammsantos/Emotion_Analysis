import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib as jb
from collections import Counter
from nltk.corpus import stopwords
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix


'''
1. Preparing data
'''
# First look of the data
data=pd.read_csv('Twitter_Emotion_Dataset.csv')
print(data.head(25))
print(data.tail(25))
print(data.columns.values)
print(data[data['label'].isnull()==True])
print(data[data['tweet'].isnull()==True])
print(data['label'].value_counts())

# removing row duplicates
grouped=data.groupby('label')
print(grouped.ngroups)
print(grouped.describe())
data.drop_duplicates(inplace=True)
data.reset_index(inplace=True)

# words length and frequencies
data['twlist']=data['tweet'].apply(lambda x: x.replace(',',' ').replace('.',' ').replace('/',' ').replace(' - ',' ').replace('?','').replace('!','').split(' ')) # column that contain list of words for each tweet

k=0
for i in data['twlist']: # loop for removing empty elements created by split
    j=0
    length = len(i)
    while j<length:
        if i[j]=='':
            data['twlist'].iloc[k].remove('')
            length-=1  
            continue
        j+=1
    k+=1

data['twlen']=data['twlist'].apply(len) # column that contain length of list for each tweet
print('The longest sentence has: '+str(data['twlen'].max())+' words')
print('\n')
data['words']=data['twlist'].apply(lambda x:' '.join(x)).apply(str.lower) # column that contain pre-processed words for each tweet for further analysis
groupedfix=data.groupby('label')

freqdata=pd.Series(' '.join(data['words']).lower().split(' ')).value_counts()[:25]
print('Most common words in tweets data: ')
print(freqdata)
print('\n')

common=[]
for i in groupedfix:
    freqdatalabel=pd.Series(' '.join(i[1]['words']).lower().split(' ')).value_counts()[:25]
    common.extend(freqdatalabel.index.values)
    print('Most common words for label '+i[0]+':')
    print(freqdatalabel)
    print('\n')

comcount=Counter(common)
comcountfix={key:val for key, val in comcount.items() if val!=1} 

stops=list(comcountfix.keys()) # setting up stopwords from the most frequent words in common amongst the labels
stops.append('[sensitive-no]')

# NOTE: Usage of combined NLTK and Sastrawi stopwords library worsen the accuracy for the prediction 
# In this case, using custom stopwords slightly improve the accuracy than not using any at all
nltkstopwds=stopwords.words('indonesian')
factory = StopWordRemoverFactory()
stopwds = factory.get_stop_words()

allstopwords=list(set(stopwords.words('indonesian')+stopwds))

'''
2. Plotting data for analysis
'''
plt.figure('Histogram for Total Words',figsize=(25,25))
plt.suptitle('Histogram for Total Words in The Tweets for Each Label',size=25)
i=1
for group in groupedfix:
    plt.subplot(2,3,i)
    plt.title(group[0])
    plt.hist(group[1]['twlen'],bins=range(80))
    i+=1    
plt.subplots_adjust(hspace=.6,wspace=.4)
plt.savefig('./histogramw.png',format='png')

plt.figure('Graph for Most Common Words',figsize=(25,25))
plt.suptitle('Graph for Most Common Words in The Tweets for Each Label',size=25)
i=1
for group in groupedfix:
    freqdatalabel=pd.Series(' '.join(group[1]['words']).lower().split(' ')).value_counts()[:10]
    plt.subplot(2,3,i)
    plt.xlabel('Words Occurrences')
    plt.ylabel('Words')
    plt.title(group[0])
    sns.barplot(x=freqdatalabel,y=freqdatalabel.index.values)
    i+=1

plt.subplots_adjust(hspace=.6,wspace=.4)
plt.savefig('./plotw.png',format='png')

plt.show()


'''
2. Establish the machine learning model
'''
x=data['tweet']
y=data['label']
xtr,xts,ytr,yts=train_test_split(x,y,test_size=.2,random_state=420)

multinomialPipeline = Pipeline([
    ('cv',CountVectorizer(stop_words=stops)),
    ('classifier',MultinomialNB())
])
multinomialPipeline.fit(xtr,ytr)
multinomialPrediksi = multinomialPipeline.predict(xts)

complementPipeline = Pipeline([
    ('cv',CountVectorizer(stop_words=stops)),
    ('classifier',ComplementNB())
])
complementPipeline.fit(xtr,ytr)
complementPrediksi = complementPipeline.predict(xts)

print('Dengan menggunakan MultinomialNB: ')

print(classification_report(yts,multinomialPrediksi))
print(confusion_matrix(yts,multinomialPrediksi))
print('\n')

print('Dengan menggunakan ComplementNB: ')
print(classification_report(yts,complementPrediksi))
print(confusion_matrix(yts,complementPrediksi))
print('\n')

# print(complementPipeline.predict(['yaelah gitu doang aja gabisa payah']))
# print(multinomialPipeline.score(xts,yts))
# print(complementPipeline.score(xts,yts))
# NOTE: From the prediction result we found that ComplementNB is slightly better in predicting the test data, therefore we are going to use it further

jb.dump(complementPipeline,'modelComplement')