import os
import pickle
import sys
import numpy
from scipy.sparse import lil_matrix
from tqdm import tqdm
import utils
from sklearn.cluster import DBSCAN

from Models.Cluster import Cluster

articles_path = utils.read_path("article-serialized-path-tfidf")

num_articles = len(os.listdir(articles_path))
with open(articles_path + str(0), "rb") as f:
    article = pickle.load(f)
    num_encoding = article.encoding.shape[1]

encodings = lil_matrix((num_articles, num_encoding), dtype=numpy.float32)

for article_num in tqdm(range(num_articles)):
    with open(articles_path + str(article_num), "rb") as f:
        article = pickle.load(f)
    encodings[article_num] = article.encoding
print("loading encodings into lil_matrix, the size is %s MB" % (sys.getsizeof(encodings) / 1024 / 1024))

dbscan = DBSCAN(metric="cosine", n_jobs=-1)
dbscan.fit(encodings)
labels = dbscan.labels_
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print("for dbscan, n_clusters is %s"%(n_clusters))
clusters = [Cluster([], i) for i in range(n_clusters)]

for i, encoding in enumerate(encodings):
    if i % 20000 == 0:
        print(i)
    cluster_num = dbscan.labels_[i]
    cluster = clusters[cluster_num]
    with open(articles_path + str(i), "rb") as f:
        article = pickle.load(f)
    cluster.articles.append(article)

Cluster.clear() # clear the files in the cluster directory
for i in range(n_clusters):
    clusters[i].serialize()

