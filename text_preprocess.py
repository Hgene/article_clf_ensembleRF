import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import nltk
from nltk.tokenize import word_tokenize
from konlpy.tag import Okt

okt = Okt()

def binary(value):
    cut_value = 30
    try:
        if type(value) is str:
            return 0
        elif float(value) >= cut_value:
            return 1
        else:
            return 0
    except TypeError:
        print(value)


def reverse(value):
    return 1-value


def to_str(token_text):
    return str(token_text).replace(',', '').replace("'", '').replace("[", '').replace("]", '').replace("▲", '')


def custom_tokenize(text):
    if not text:
        print('The text to be tokenized is a None type. Defaulting to blank string.')
        text = ''
    return okt.nouns(text)


def load_stopwords(dir_, custom_words):
    stop_words=''
    with open(dir_,'r', encoding='utf-8') as f:
        stop_words = f.readlines()
    stop_words = [str(x[:len(x)-1]) for x in stop_words]
    stop_words.extend(custom_words)
    return stop_words


def remove_stop_words(word_tokens):
    removed_tokens=[]
    for w in word_tokens:
        if w not in stop_words:
            removed_tokens.append(w)
    return removed_tokens



news = pd.read_csv('./data/all_articles.csv', header=0, encoding = 'utf_8_sig')

cols = ['MainText','Title','보건', '축산업', '수산양식', '농업', '산업', '교통', '전력', '기타피해', '긍정', '정책', '비관련']
news= news[cols].fillna(0)
news['축산업']= news['축산업'].replace(' ', 0)
news['전력']= news['전력'].replace('누진제', 0)

labels = ['보건', '축산업', '수산양식', '농업', '산업', '교통', '전력', '기타피해', '긍정', '정책', '비관련']
news[labels] = news[labels].astype(int)
for label_i in labels:
    news[label_i] = news[label_i].apply(binary)
news['비관련'] = news['비관련'].apply(reverse)
news = news[news['비관련']==1]

custom_words = [',','‘','’','[',']','(',')','"','`','``','','...', "''", "'", "''", '\n', '\t', '.', '“','”','고','「','▲']
stop_word_dir = './data/stop_words_korean.txt'
stop_words = load_stopwords(stop_word_dir, custom_words)

news['MainText'] = news['MainText'].apply(custom_tokenize).apply(remove_stop_words).apply(to_str)
news['MainText'].iloc[1]


news.reset_index().to_feather('./train_set')
del news
train_set = pd.read_feather('./train_set')

#article load and save dataset, You can skip this cell if you have test_set file
texts_list = [int(x[:-4]) for x in os.listdir('../articles/all_articles')]

test_set = pd.DataFrame(np.zeros((len(texts_list),2)), columns=['MainText', 'Identifier'])
err_idx = []

for i, text_id in enumerate(texts_list):
    if i %10000 == 0:
        print(i)
    try:
        with open("../articles/all_articles/{}.txt".format(text_id), encoding='UTF-8') as f:
            data = f.read()
            data = word_tokenize(data)
            data = remove_stop_words(data)
            data = to_str(data)
            test_set['MainText'].loc[i] = data
            test_set['Identifier'].loc[i] = text_id
    except FileNotFoundError:
        err_idx.append(text_id)


test_set.to_feather('./test_set')
#test_set_copy = pd.read_feather('./test_set')
#test_set_copy.head()
