import pickle
import unittest
from utils import *
from gensim.utils import simple_preprocess
from OutlierDetection.outlier_detection import outlierdetecion

class MyTestCase(unittest.TestCase):
    def test_json_to_csv(self):
        csv_file = ".\data\Sarcasm_Headlines_Dataset_v2.csv"
        json_file = ".\data\Sarcasm_Headlines_Dataset_v2.json"
        rc = json_to_csv(json_file, csv_file)
        self.assertEqual(rc, 0)

    def test_read_corpus(self):
        article_path = "../serialized_articles_tfidf/"
        test_article_indices = [0, 1, 28618]
        test_articles = ["thirtysomething scientists unveil doomsday clock of hair loss",
                         '"dem rep. totally nails why congress is falling short on gender, racial equality"',
                         "dad clarifies this not a food stop"]
        for i, id in enumerate(test_article_indices):
            with open(article_path + str(id), "rb") as f:
                article = pickle.load(f)
                self.assertEqual(article.title, " ".join(simple_preprocess(test_articles[i])))

    def test_longest_common_subsequence(self):
        self.assertEqual(longest_common_subsequence("", "1"), 0)
        self.assertEqual(longest_common_subsequence("a b c", "c"), 1)
        self.assertEqual(longest_common_subsequence("a b c", "a"), 1)
        self.assertEqual(longest_common_subsequence("s s s", "s s"), 2)

    def test_cluster(self):
        cluster_paths = [read_path("cluster-serialized-path-kmeans"),
                         read_path("cluster-serialized-path-dbscan")]
        for cluster_path in cluster_paths:
            self.assertNotEqual(os.listdir(cluster_path) , 0)
            for cluster_name in os.listdir(cluster_path):
                with open(os.path.join(cluster_path, cluster_name), "rb") as f:
                    cluster = pickle.load(f)
                    self.assertNotEqual(len(cluster.articles), 0)
                    self.assertNotEqual(len(cluster.center), 0)

    def test_outlier_detection(self):
        methods = ["LOF", "text"]
        cluster_path = read_path("cluster-serialized-path-kmeans")
        self.assertNotEqual(len(os.listdir(cluster_path)), 0)
        cluster_file_names = os.listdir(cluster_path)
        for cluster_file_name in cluster_file_names:
            with open(cluster_path + "/" + cluster_file_name, "rb") as f:
                cluster = pickle.load(f)
                for method in methods:
                    outliers = outlierdetecion(cluster, method)
                    self.assertIsInstance(outliers, list)

    def test_something(self):
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
