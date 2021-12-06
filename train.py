import sys
import json
import re
import numpy as np
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SQLContext, Row
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from sklearn.feature_extraction.text import HashingVectorizer

from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml.classification import LogisticRegression

from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn.cluster import MiniBatchKMeans

import pickle

# global variables
first_batch = True
tweet_id = 0
sc = None
perceptron_train_model=None
bernoulli_train_model=None
sgd_classifier_train_model=None
mini_batch_kmeans_cluster_train_model=None
allow_flag = False


def p_process(rdd):
    global tweet_id
    global first_batch
    global sc
    global perceptron_train_model
    global bernoulli_train_model
    global sgd_classifier_train_model
    global mini_batch_kmeans_cluster_train_model
    global allow_flag

    start_index = 0
    list1 = []
    tweet_list = []
    sentiments = []
    col = ["sentiment","tweet"]
    if(not rdd.isEmpty()):
        if first_batch:
            first_batch = False
            start_index = 1
        row = rdd.map(lambda x: Row(sentiment=x[0], tweet=x[1]))
        a = rdd.collect()[start_index:]
        a = list(map(lambda i: i.split(",", 1), a))
        for i in range(len(a)):
            sentiment = int(a[i][0])
            tweet = a[i][1].lstrip().rstrip()
            tweet = re.sub(r'http\S+', '', tweet)
            tweet = tweet.strip('\"').lstrip().rstrip()
            tweet = " ".join(
                filter(lambda x: x[0] != '@', tweet.split())).lstrip().rstrip()
            tweet = " ".join(
                filter(lambda x: x[0] != '#', tweet.split())).lstrip().rstrip()

            # Removing punctuations in string using regex
            tweet = re.sub(r'[^\w\s]', ' ', tweet).lstrip().rstrip()

            # Converting multiple white-spaces into single whitespace
            tweet = re.sub(r' +', ' ', tweet).lstrip().rstrip()

            # converting tweet to lowercase
            tweet = tweet.lower()
            tweet_list.append(tweet)
            sentiments.append(sentiment)

            if len(tweet) != 0:
                list1.append((sentiment, tweet))
                # print(tweet)

        sqlContext = SQLContext(sc)
        df = sqlContext.createDataFrame(list1, col)

        vectorizer=HashingVectorizer(n_features=1000)

        X_train=vectorizer.fit_transform(tweet_list)
        y_train= sentiments

        if allow_flag:
            percetron_train_model = Perceptron()
            bernoulli_train_model = BernoulliNB()
            sgd_classifier_train_model = SGDClassifier()
            mini_batch_kmeans_cluster_train_model = MiniBatchKMeans(n_clusters=2)

            percetron_train_model.partial_fit(X_train,y_train, classes = [0,4])
            bernoulli_train_model.partial_fit(X_train,y_train, classes = [0,4])
            sgd_classifier_train_model.partial_fit(X_train,y_train, classes = [0,4])
            mini_batch_kmeans_cluster_train_model.partial_fit(X_train,y_train)
        else:
            allow_flag = True


sc = SparkContext(master="local[2]", appName="mlss_bd_198_200_367_503")
ssc = StreamingContext(sc, 1)

lines = ssc.socketTextStream("localhost", 6100)

perceptron_dbfile = open('perceptron', 'wb')
bernoulli_dbfile = open('bernoulli', 'wb')
sgd_classifier_dbfile = open('sgd_classifier', 'wb')
mini_batch_kmeans_cluster_dbfile = open('mini_batch_kmeans_cluster', 'wb')

# my_schema = StructType([StructField(name='tweet', dataType=StringType(), nullable=True), StructField(name='sentiment', dataType=IntegerType(), nullable=True)])

row = lines.flatMap(lambda line: json.loads(line))

row.foreachRDD(p_process)

perceptron_db = perceptron_train_model
bernoulli_db = bernoulli_train_model
sgd_classifier_db = sgd_classifier_train_model
mini_batch_kmeans_cluster_db = mini_batch_kmeans_cluster_train_model

pickle.dump(perceptron_db,perceptron_dbfile)
pickle.dump(bernoulli_db,bernoulli_dbfile)
pickle.dump(sgd_classifier_db,sgd_classifier_dbfile)
pickle.dump(mini_batch_kmeans_cluster_db,mini_batch_kmeans_cluster_dbfile)

perceptron_dbfile.close()
bernoulli_dbfile.close()
sgd_classifier_dbfile.close()
mini_batch_kmeans_cluster_dbfile.close()

ssc.start()
ssc.awaitTermination()
ssc.stop()
