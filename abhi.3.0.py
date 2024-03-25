#importing required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import string
from nltk.tokenize import word_tokenize
nltk.download('punkt')#dataset: https://www.kaggle.com/karthickveerakumar/spam-filter
emails = pd.read_csv('/content/sample_data/emails.csv')
emailsemails.info()
emails = emails.drop_duplicates(keep = 'last') #remove all duplicate emails from the dataframe
emails
#data visualization using matplotlib
emails.spam.value_counts().plot(kind='pie',
                                    explode=[0,.1],
                                    figsize=(6,6),
                                    autopct='%.2f%%')
plt.title('Normal Mails vs Spam mails')
plt.legend(['Normal','Spam'])
plt.show()
emails.spam.value_counts()
# allocating data to the variables
spam_messages = emails[emails['spam']==1]['text']
notspam_messages = emails[emails['spam']==0]['text']
spam_words = []
notspam_words = []
#creating a function for tokenizing the text using nltk
def tokenize_spam_words(text):
    words = [w.lower() for w in word_tokenize(text) if w.lower() not in stopwords.words('english') and w.lower().isalpha()]
    spam_words.extend(words)

def tokenize_notspam_words(text):
    words = [w.lower() for w in word_tokenize(text) if w.lower() not in stopwords.words('english') and w.lower().isalpha()]
    notspam_words.extend(words)
#tokenizing the spam messages
spam_messages.apply(tokenize_spam_words)
print(spam_words[:100])

import nltk
nltk.download('stopwords')
#tokenizing the not spam messages
notspam_messages.apply(tokenize_notspam_words)
print(notspam_words[:100])
#stemming
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
# creating a function for stemming the words
def cleanup_text(message):
    message = message.translate(str.maketrans('','',string.punctuation))
    words = [stemmer.stem(w) for w in message.split() if w.lower() not in stopwords.words('english') ]
    return ' '.join(words)
emails.text = emails.text.apply(cleanup_text)
emails.head()
#feautre extraction using count vectorizer
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(stop_words = 'english')
features = vect.fit_transform(emails.text)
features.shape
# saving the feautures using the pickle
import pickle

with open('count_vectorizer.pkl','wb') as f:
    pickle.dump(vect,f)
print('done')
# data preprocessing for training the model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
#labeling the data
category =  LabelEncoder()
emails.spam = category.fit_transform(emails.spam)
emails.head()
#splitting the data into training and testing data
x_train, x_test, y_train,y_test = train_test_split(features.toarray(), emails.spam,test_size=.2)
#creating a machine learning model
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
model = GaussianNB()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
#confusion matrix
confusion_matrix(y_test,y_pred)#saving the builded model using pickle
import pickle
with open('spam_classifier.pkl','wb') as f:
    pickle.dump(model,f)
print('done')