import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from gensim.models.doc2vec import Doc2Vec

import gensim

stemmer = SnowballStemmer("english")


class Classifier:
    def __init__(self, df, test_split=0.99, doc2vec=False):
        self.df = df
        self.d2v = doc2vec
        self.df['tokens'] = self.df.text.apply(gensim.utils.simple_preprocess)
        self.train, self.test = train_test_split(df, test_size=test_split, shuffle=True)

        self.vec_model = self.set_vectorizer()

        self.vec_train, self.vec_test = self.transform_text()

        self.log_model = LogisticRegression(max_iter=300).fit(X=self.vec_train, y=self.train.positive)

    def get_accuracy(self):
        preds_train, preds_test = self.log_model.predict(self.vec_train), self.log_model.predict(
            self.vec_test)
        return accuracy_score(self.train.positive, preds_train), accuracy_score(self.test.positive, preds_test)

    def set_vectorizer(self):
        if self.d2v:
            train_tokens = list(self.df.tokens)  # training on whole dataset. otherwise would be train.tokens
            # test_corpus = list(self.test.tokens)
            train_corpus = [gensim.models.doc2vec.TaggedDocument(train_tokens[i], [i]) for i in
                            range(len(train_tokens))]

            model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)
            model.build_vocab(train_corpus)
            model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

            return model
        else:
            vectorizer =  TfidfVectorizer(min_df=20,
                                   max_df=.6,
                                   max_features=2000,
                                   use_idf=True,
                                   stop_words='english')
            return vectorizer.fit(self.df.text)


    def transform_text(self):
        if self.d2v:
            train = [self.vec_model.infer_vector(line) for line in list(self.train.tokens)]
            test = [self.vec_model.infer_vector(line) for line in list(self.test.tokens)]
        else:
            train = self.vec_model.transform(self.train.text)
            test = self.vec_model.transform(self.test.text)

        return train, test




if __name__ == '__main__':
    movies = pd.read_csv('./sentiment/movies.csv')
    yelp = pd.read_csv('./sentiment/yelps.csv')

    movies_sample = movies.sample(frac=0.1) #subset for testing, switch to 1 for final model

    test_split = [0.99, 0.98, 0.96, 0.88, 0.80, 0.75, 0.67]
    for ts in test_split:

        simple_classifier = Classifier(movies_sample,test_split=ts)
        print(simple_classifier.get_accuracy())

        d2v_classifier = Classifier(movies_sample, test_split=ts, doc2vec=True)
        print(d2v_classifier.get_accuracy())
