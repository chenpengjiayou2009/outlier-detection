import os
import pickle
import sys
import matplotlib.pyplot as plt
import numpy
from scipy.sparse import lil_matrix
from sklearn.cluster import KMeans
from tqdm import tqdm
import utils

from Models.Cluster import Cluster
from Models.Article import Article

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
print("loading encodings into lil_matrix, the size is %s MB"%(sys.getsizeof(encodings) / 1024 / 1024))
wcss = []
for i in tqdm(range(1, 11)):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(encodings)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.savefig('elbow.png')
plt.show()

# pick the appropriate cluster count
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, max_iter=300, n_init=10, random_state=0)
kmeans.fit(encodings)
clusters = [Cluster([], i) for i in range(n_clusters)]

for i, encoding in enumerate(encodings):
    if i % 20000 == 0:
        print(i)
    cluster_num = kmeans.labels_[i]
    cluster = clusters[cluster_num]
    with open(articles_path + str(i), "rb") as f:
        article = pickle.load(f)
    cluster.articles.append(article)

Cluster.clear() # clear the files in the cluster directory
for i in range(n_clusters):
    clusters[i].center = kmeans.cluster_centers_[i]
    clusters[i].serialize()

