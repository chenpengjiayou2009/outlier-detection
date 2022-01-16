class Article:
    def __init__(self, encoding, title,):
        self.encoding = encoding
        self.title = title
        self.outlier_score = 0

    def __lt__(self, other):
        return self.outlier_score > other.outlier_score