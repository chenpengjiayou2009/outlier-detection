from typing import List

from sklearn.neighbors import LocalOutlierFactor
from tqdm import tqdm

from Models.Article import Article
from Models.Cluster import Cluster
import utils
import os
import pickle


def outlierdetecion(cluster: Cluster, method: str) -> List[Article]:
    res = []
    encodings = []
    articles = cluster.articles
    print("this cluster has %s articles\n" % len(articles))
    for article in tqdm(articles):
        encodings.append(article.encoding.toarray()[0])

    if method == "LOF":
        print("please input the number of nearest neighbors to get the outliers")
        k = int(input())
        clf = LocalOutlierFactor(n_neighbors=k, algorithm="auto", n_jobs=-1)
        x = clf.fit_predict(encodings)
        for i, outlier_factor in enumerate(x):
            if outlier_factor == -1:
                print("find outlier factor == -1")
                res.append(articles[i])

            if len(res) == 10:
                break

    if method == "text":
        most_frequent_words = cluster.most_frequent_words()
        for article in articles:
            outlier_score = 0
            for word in article.title.split():
                if word in most_frequent_words:
                    outlier_score += 1
        articles.sort(reverse=True)
        res = articles[:10]

    cluster.outliers = res
    cluster.serialize()

    return res

def main():
    method = input("please input method to find outlier detection, LOF or text")
    cluster_path = utils.read_path("cluster-serialized-path-kmeans")
    cluster_file_names = os.listdir(cluster_path)
    for i, cluster_file_name in enumerate(cluster_file_names):
        print("now processing " + str(i) + " cluster")
        with open(os.path.join(cluster_path, cluster_file_name), "rb") as f:
            cluster = pickle.load(f)
            outliers = outlierdetecion(cluster, method)
            cluster.outliers = outliers
            cluster.serialize()

if __name__ == "__main__":
    main()