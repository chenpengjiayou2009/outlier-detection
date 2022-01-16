import os
from collections import defaultdict
from typing import List
from Models.Article import Article
import utils
import pickle
cluster_algorithm = input("please choose cluster algorithm, kmeans or dbscan")
if cluster_algorithm == "kmeans":
    cluster_path = "cluster-serialized-path-kmeans"
else:
    cluster_path = "cluster-serialized-path-dbscan"

serialization_path = utils.read_path(cluster_path)

class Cluster:
    def __init__(self, articles: List[Article], cluster_index:int):
        self.articles = articles
        self.center = []
        self.serialization_path = serialization_path
        self.outliers = []
        self.cluster_index = cluster_index

    def serialize(self):
        with open(self.serialization_path + str(self.cluster_index), "wb") as f:
            pickle.dump(self, f)

        print(str(self.cluster_index) + " cluster serialized finished")

    def most_frequent_words(self) -> dict:
        frequency = defaultdict(int)
        for article in self.articles:
            if isinstance(article.title, list):
                for word in article.title:
                    frequency[word] += 1
            else:
                for word in article.title.split():
                    frequency[word] += 1

        return dict(list(sorted(frequency.items(), key=lambda x: -x[1]))[:10])

    def clear(self):
        os.remove()

    @staticmethod
    def clear():
        for file in os.listdir(serialization_path):
            os.remove(serialization_path + file)
