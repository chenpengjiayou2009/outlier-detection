import json
import os
import sys

import numpy
import pandas as pd
import gensim
import smart_open
from typing import List
from scipy import spatial
from configparser import ConfigParser
from scipy.sparse import csr_matrix

def cosine_similarity(a:List[float], b:List[float]) -> float:
    if isinstance(a, numpy.ndarray) or isinstance(b, numpy.ndarray):
        a = a.tolist()
    if isinstance(b, csr_matrix) or isinstance(a, csr_matrix):
        b = b.toarray().tolist()[0]

    return spatial.distance.cosine(a, b)

def read_path(param:str):
    config = ConfigParser()
    dir_name = "."
    config_path = os.path.join(dir_name, "config.ini")
    if os.path.exists(config_path):
        config.read(config_path)
    else:
        dir_name = ".."
        config_path = os.path.join(dir_name, "config.ini")
        config.read(config_path)

    path = os.path.join(dir_name, config.get("paths", param))

    # if os.path.abspath("..").split("\\")[-2] == "outlier_detection":
    #     path = os.path.join("..", path)

    if not os.path.exists(path):
        os.mkdir(path)
    return path

def read_corpus(fname, tokens_only=False):
    if fname.endswith(".csv"):
        data_frame = pd.read_csv(fname)
        for i, line in enumerate(data_frame["headline"]):
            tokens = gensim.utils.simple_preprocess(line)
            if tokens_only:
                yield " ".join(tokens)
            else:
                yield gensim.models.doc2vec.TaggedDocument(tokens, [i])
    elif fname.endswith(".cor"):
        with smart_open.open(fname, encoding="iso-8859-1") as f:
            for i, line in enumerate(f):
                tokens = gensim.utils.simple_preprocess(line)
                if tokens_only:
                    yield tokens
                else:
                    # For training data, add tags
                    yield gensim.models.doc2vec.TaggedDocument(tokens, [i])

def json_to_csv(json_file:str, csv_file:str) -> int:
    csv_data = pd.DataFrame(columns=["headline"])
    f = open(json_file, "r")
    line = f.readline()
    while line:
        news = json.loads(line)
        headline = news["headline"]
        csv_data = csv_data.append({"headline": headline}, ignore_index=True)
        line = f.readline()

    csv_data.to_csv(csv_file)
    f.close()
    if csv_data.shape != (0,1):
        return 0
    else:
        return 1

def longest_common_subsequence(s1:str, s2:str) -> int:
    words_s1 = s1.split()
    words_s2 = s2.split()
    len1 = len(words_s1)
    len2 = len(words_s2)
    dp = [[0 for _ in range(len2+1)]for _ in range(len1+1)]

    for i in range(1, len1+1):
        for j in range(1, len2+1):
            if words_s1[i-1] == words_s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    return dp[-1][-1]

