import matplotlib.pyplot as plt
import sys
import csv

files = ['perceptron_score.csv', 'bernoulli_score.csv', 'sgd_classifier_score.csv', 'mini_batch_kmeans_cluster_score.csv']
  
x = []
y = []

for file in files:
    with open(file,'r') as csvfile:
        plots = csv.reader(csvfile, delimiter = ',')
        for row in plots:
            x.append(int(row[0]))
            y.append(float(row[1]))
      
    plt.bar(x, y, color = 'g', width = 0.72)
    plt.xlabel("Batch")
    plt.ylabel("Score")
    plt.title("Variation of scores: {} classifier".format(file.split("_")[0]))
    plt.legend()
    plt.show()
