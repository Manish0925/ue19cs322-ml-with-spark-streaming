import csv
import json
import pickle
import re
from sklearn.feature_extraction.text import HashingVectorizer
from pyspark import SparkContext
from pyspark.sql import Row
from pyspark.streaming import StreamingContext

# global variables
first_batch = True
tweet_id = 0
sc = None
perceptron_train_model = None
bernoulli_train_model = None
sgd_classifier_train_model = None
mini_batch_kmeans_cluster_train_model = None

def p_process(rdd):
    global tweet_id, first_batch, sc, perceptron_train_model, bernoulli_train_model, sgd_classifier_train_model, mini_batch_kmeans_cluster_train_model, field, batch_no
    start_index = 0
    list1 = []
    tweet_list = []
    sentiments = []
    if not rdd.isEmpty():
        if first_batch:
            first_batch = False
            start_index = 1
        a = rdd.collect()[start_index:]
        a = list(map(lambda i: i.split(",", 1), a))
        for i in range(len(a)):
            sentiment = int(a[i][0])
            tweet = a[i][1].lstrip().rstrip()
            tweet = re.sub(r"http\S+", "", tweet)
            tweet = tweet.strip('"').lstrip().rstrip()
            tweet = (
                " ".join(filter(lambda x: x[0] != "@", tweet.split())).lstrip().rstrip()
            )
            tweet = (
                " ".join(filter(lambda x: x[0] != "#", tweet.split())).lstrip().rstrip()
            )

            # Removing punctuations in string using regex
            tweet = re.sub(r"[^\w\s]", " ", tweet).lstrip().rstrip()

            # Converting multiple white-spaces into single whitespace
            tweet = re.sub(r" +", " ", tweet).lstrip().rstrip()

            # converting tweet to lowercase
            tweet = tweet.lower()
            tweet_list.append(tweet)
            sentiments.append(sentiment)

            if len(tweet) != 0:
                list1.append((sentiment, tweet))

        vectorizer = HashingVectorizer(n_features=1000)

        X_test = vectorizer.fit_transform(tweet_list)
        y_test = sentiments

        with open("perceptron_score.csv", "a") as percepton_score_file:
            row = [[batch_no, perceptron_train_model.score(X_test, y_test)]]
            csvwriter = csv.writer(percepton_score_file)
            csvwriter.writerows(row)

        with open("bernoulli_score.csv", "a") as bernoulli_score_file:
            row = [[batch_no, bernoulli_train_model.score(X_test, y_test)]]
            csvwriter = csv.writer(bernoulli_score_file)
            csvwriter.writerows(row)

        with open("sgd_classifier_score.csv", "a") as sgd_classifier_score_file:
            row = [[batch_no, sgd_classifier_train_model.score(X_test, y_test)]]
            csvwriter = csv.writer(sgd_classifier_score_file)
            csvwriter.writerows(row)

        with open(
            "mini_batch_kmeans_cluster_score.csv", "a"
        ) as mini_batch_kmeans_cluster_score_file:
            row = [
                [batch_no, mini_batch_kmeans_cluster_train_model.score(X_test, y_test)]
            ]
            csvwriter = csv.writer(mini_batch_kmeans_cluster_score_file)
            csvwriter.writerows(row)

    batch_no += 1

sc = SparkContext(master="local[2]", appName="mlss_bd_198_200_367_503")
ssc = StreamingContext(sc, 1)

lines = ssc.socketTextStream("localhost", 6100)

with open("perceptron", "rb") as perceptron_dbfile:
    perceptron_train_model = pickle.load(perceptron_dbfile)
with open("bernoulli", "rb") as bernoulli_dbfile:
    bernoulli_train_model = pickle.load(bernoulli_dbfile)
with open("sgd_classifier", "rb") as sgd_classifier_dbfile:
    sgd_classifier_train_model = pickle.load(sgd_classifier_dbfile)
with open("mini_batch_kmeans_cluster", "rb") as mini_batch_kmeans_cluster_dbfile:
    mini_batch_kmeans_cluster_train_model = pickle.load(
        mini_batch_kmeans_cluster_dbfile
    )

row = lines.flatMap(lambda line: json.loads(line))
row.foreachRDD(p_process)
field = ["Batch", "Score"]
batch_no = 1

ssc.start()
ssc.awaitTermination()
ssc.stop()
