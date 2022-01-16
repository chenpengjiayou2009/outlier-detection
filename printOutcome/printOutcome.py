import os
import pickle

from Models import Cluster
import utils

cluster_path = utils.read_path("cluster-serialized-path-kmeans")
clusters = list(os.listdir(cluster_path))
outcome_file = open("./outcome.txt", "w")
for i, cluster_hash in enumerate(clusters):
    with open(os.path.join(cluster_path, cluster_hash), 'rb') as f:
        cluster = pickle.load(f)
        outcome_file.write("cluster " + str(i) + ": \n")
        outcome_file.write("articles: ")
        for article in cluster.articles[:20]:
            outcome_file.write(article.title + "\n")
        outcome_file.write("outliers: \n")
        for outlier in cluster.outliers:
            outcome_file.write(outlier.title + "\n")
        outcome_file.write("\n \n")
        outcome_file.write("")

outcome_file.close()

