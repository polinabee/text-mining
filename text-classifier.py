import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

stemmer = SnowballStemmer("english")


class Classifier:
    def __init__(self, df_a, df_b, subset, vocab=None):
        self.data_a = df_a.sample(frac=subset)
        self.data_b = df_b.sample(frac=subset)
        self.vocab = vocab
        self.vectorizer = self.set_vectorizer()
        self.vec_fit = self.vectorizer.fit(self.data_a.text)
        self.vec_trfm_a = self.vec_fit.transform(self.data_a.text)
        self.vec_trfm_b = self.vec_fit.transform(self.data_b.text)
        self.log_model = LogisticRegression(max_iter=300).fit(X=self.vec_trfm_a, y=self.data_a.positive)

    def get_accuracy(self):
        preds_a, preds_b = self.log_model.predict(self.vec_trfm_a), self.log_model.predict(self.vec_trfm_b)
        return accuracy_score(self.data_a.positive, preds_a), accuracy_score(self.data_b.positive, preds_b)

    def set_vectorizer(self):

        if self.vocab is not None:
            return TfidfVectorizer(vocabulary=self.vocab)
        else:
            return TfidfVectorizer(min_df=20,
                                   max_df=.6,
                                   max_features=2000,
                                   use_idf=True,
                                   stop_words='english')


def preprocess(doc):
    not_alphanumeric_or_space = re.compile(r'[^\w|\s]')
    doc = re.sub(not_alphanumeric_or_space, '', doc)
    words = [word.lower() for word in doc.split() if word not in stopwords.words('english')]
    #     words = [stemmer.stem(word.lower()) for word in doc.split() if word not in stopwords.words('english')]
    return words


def word_count(doc):
    counts = {}
    for word in doc:
        try:
            counts[word] += 1
        except KeyError:
            counts[word] = 1
    return counts


def get_vocab(df, n):
    df_head = df.head(n)
    preprocessed = df_head.text.apply(preprocess)
    return word_count([word for line in list(preprocessed) for word in line])


def get_intersection(data_a, data_b, lines):
    a_vocab = get_vocab(data_a, lines)
    b_vocab = get_vocab(data_b, lines)
    return a_vocab.keys() & b_vocab.keys()


if __name__ == '__main__':
    movies = pd.read_csv('./sentiment/movies.csv')
    yelp = pd.read_csv('./sentiment/yelps.csv')

    lim = .3

    my_classifier = Classifier(movies, yelp, lim)
    print(my_classifier.get_accuracy())

    ym_classifier = Classifier(yelp, movies, lim)
    print(ym_classifier.get_accuracy())

    intersection = get_intersection(my_classifier.data_a, my_classifier.data_b, 300)
    print

    my_classifier_int = Classifier(movies, yelp, lim, intersection)
    print(my_classifier.get_accuracy())

    ym_classifier_int = Classifier(yelp, movies, lim, intersection)
    print(ym_classifier.get_accuracy())
