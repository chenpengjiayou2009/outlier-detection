import datetime
import os
import pickle
from tqdm import tqdm
import gensim
from gensim import models

import utils
from Models.Article import *

train_file = "../data/Sarcasm_Headlines_Dataset_v2.csv"
test_file = ""
train_corpus = list(utils.read_corpus(train_file))
# # test_corpus = list(read_corpus(test_file, tokens_only=True))
#
# train the model
train_model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)
train_model.build_vocab(train_corpus)
train_start_time = datetime.datetime.now()
print("start to train corpus")
train_model.train(train_corpus, total_examples=train_model.corpus_count, epochs=train_model.epochs)
print("training end, training takes %s seconds"%((datetime.datetime.now() - train_start_time).total_seconds()))
train_model_path = os.path.join(utils.read_path("train-model-path-doc2vec"), "train_model_doc2vec")
train_model.save(train_model_path)
print("model saved")

# load trained model
train_model = gensim.models.doc2vec.Doc2Vec.load(train_model_path)

# assessing the model
ranks = []
second_ranks = []

# serialize articles
articles_folder = utils.read_path("article-serialized-path-doc2vec")
for doc_id in tqdm(range(len(train_corpus))):
    inferred_vector = train_model.infer_vector(train_corpus[doc_id].words)
    article = Article(inferred_vector, train_corpus[doc_id].words)
    with open(articles_folder + str(doc_id), "wb") as f:
        pickle.dump(article, f)
    # assess the model
    # sims = train_model.dv.most_similar([inferred_vector], topn=len(train_model.dv))
    # rank = [docid for docid, sim in sims].index(doc_id)
    # ranks.append(rank)
    # second_ranks.append(sims[1])



