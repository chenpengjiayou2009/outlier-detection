import os
import pickle
from tqdm import tqdm
import utils
from sklearn.feature_extraction.text import TfidfVectorizer
from Models.Article import Article

train_file = "../data/Sarcasm_Headlines_Dataset_v2.csv"
test_file = ""
train_corpus = list(utils.read_corpus(train_file, tokens_only=True))
tf_idf_matrix = TfidfVectorizer().fit_transform(train_corpus)
# serialize articles
articles_folder = utils.read_path("article-serialized-path-tfidf")
for doc_id in tqdm(range(len(train_corpus))):
    inferred_vector = tf_idf_matrix[doc_id]
    article = Article(encoding=inferred_vector, title=train_corpus[doc_id])
    if not os.path.exists(articles_folder):
        os.mkdir(articles_folder)
    with open(articles_folder + str(doc_id), "wb") as f:
        pickle.dump(article, f)