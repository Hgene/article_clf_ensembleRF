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


import random



def build_vocab(topic):
    key_sentences = []
    with open('./data/{}.txt'.format(topic), 'r', encoding='utf_8_sig') as f:
        data = f.readlines()
        data = [x.replace('\n', '') for x in data]
        key_sentences.extend(data)

    key_sentences = list(set(key_sentences))
    print(key_sentences[-2:])
    vec = CountVectorizer()
    vec.fit_transform(key_sentences)

    return vec


def confusion_mat(predict, actual):
    accuracy = np.sum(predict == actual) / len(actual)

    # precision=P(A==1|P==1)
    precision = np.sum((predict == 1) & (actual == 1)) / np.sum(predict == 1)

    # recall=P(P==1|actual==1)
    recall = np.sum((predict == 1) & (actual == 1)) / np.sum(actual == 1)
    f1_score = 2 * recall * precision / (recall + precision)

    return accuracy, precision, recall, f1_score


def run(train, train_y, test, test_y, tree_id, name):
    RF = RandomForestClassifier(n_estimators=5)
    RF.fit(train, train_y)
    pred_y = RF.predict(test)
    accuracy, precision, recall, f1_score = confusion_mat(pred_y, test_y)

    return pred_y, [accuracy, precision, recall, f1_score]


news = pd.read_feather('./train_set')

topics = []
for file in os.listdir('./data'):
    if '.txt' in file:
        topics.append(file[:-4])
topics.remove('stop_words_korean')
print(topics)

news_train, news_test = train_test_split(news, test_size=0.3, random_state=42)

for topic in topics:

    with open('./val_result/train_{}.csv'.format(topic), 'w', encoding='utf_8_sig') as f:
        f.write('Classifier, TDMatrix, accuracy, precision, recall, f1_score\n')

    topic_len = len(news_train[news_train[topic] == 1])
    print("topic_{}_len : {}".format(topic, topic_len))
    vec = build_vocab(topic)

    test = vec.transform(list(news_test['MainText'])).toarray()
    test_y = np.array(news_test[topic]).reshape(-1, )  # 1)

    vote1 = np.zeros(len(news_test))
    vote2 = np.zeros(len(news_test))
    precision1 = 0
    precision2 = 0

    for i in range(200):
        idx = []
        idx.extend(list(random.sample(list(news_train[news_train[topic] == 0].index), topic_len)))
        idx.extend(list(news_train[news_train[topic] == 1].index))

        body = list(news_train.loc[idx, :]['MainText'])
        label = np.array(list(news_train.loc[idx, :][topic])).reshape(-1, )  # 1)

        X = vec.transform(body).toarray()
        countMatrix = X
        tf_idf = TfidfTransformer(smooth_idf=True, use_idf=True)
        tfidfMatrix = tf_idf.fit_transform(X).toarray()

        pred_y1, result1 = run(countMatrix, label, test, test_y, 'Count', i)
        pred_y2, result2 = run(tfidfMatrix, label, test, test_y, 'TF-IDF', i)

        vote1 += np.array(pred_y1) * result1[1]
        precision1 += result1[1]
        vote2 += np.array(pred_y2) * result2[1]
        precision2 += result2[1]

    pred_y1 = np.around(vote1 / precision1).reshape(-1, )  # 1)
    pred_y2 = np.around(vote2 / precision2).reshape(-1, )  # 1)

    accuracy, precision, recall, f1_score = confusion_mat(pred_y1, test_y)
    with open('./val_result/train_{}.csv'.format(topic), 'a', encoding='utf_8_sig') as f:
        f.write('{}, {}, {}, {}, {}, {}\n'.format('RF', 'Count', accuracy, precision, recall, f1_score))

    accuracy, precision, recall, f1_score = confusion_mat(pred_y2, test_y)
    with open('./val_result/train_{}.csv'.format(topic), 'a', encoding='utf_8_sig') as f:
        f.write('{}, {}, {}, {}, {}, {}\n'.format('RF', 'TF-IDF', accuracy, precision, recall, f1_score))

    print(confusion_matrix(test_y, pred_y1))
    print(confusion_mat(pred_y1, test_y))
    print(confusion_matrix(test_y, pred_y2))
    print(confusion_mat(pred_y2, test_y))
