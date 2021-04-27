import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")


class Vectorizer():
    def __init__(self, docs):
        self.docs = [self.preprocess(line) for line in docs]
        self.vocab_count = self.word_count(' '.join(self.docs))
        self.reduced_vocab_count = self.limited_vocab()
        self.term_freq_mtx = np.array(
            [[line.count(word) / len(line) for word in self.reduced_vocab_count.keys()] for line in docs])
        self.inv_doc_freq = [self.inverse_doc_freq(word) for word in self.reduced_vocab_count.keys()]
        self.tfidf_mtx = np.array([[x / np.linalg.norm(x)] for x in (self.term_freq_mtx * self.inv_doc_freq)])
        self.reweighted_mtx = self.tfidf_mtx * self.weights()

    def word_count(self, doc):
        counts = {}
        for word in doc.split():
            try:
                counts[word] += 1
            except KeyError:
                counts[word] = 1
        return counts

    def preprocess(self, doc):
        not_alphanumeric_or_space = re.compile(r'[^\w|\s]')
        doc = re.sub(not_alphanumeric_or_space, '', doc)
        # words = [lemmatizer.lemmatize(word) for word in doc.split()]
        words = [stemmer.stem(word) for word in doc.split() if word not in stopwords.words('english')]
        return ' '.join(words).lower()

    def inverse_doc_freq(self, term):
        df = sum([1 for doc in self.docs if term in doc])
        return np.log(len(self.docs) / df)

    def limited_vocab(self):
        term_popularity = {term: sum([1 for doc in self.docs if term in doc]) for term in self.vocab_count.keys()}
        return {key: value for key, value in self.vocab_count.items() if term_popularity[key] > 1}

    def weights(self):
        POS = nltk.pos_tag([key for key in self.reduced_vocab_count.keys()])
        low = ['DT']
        med = ['PRP']
        high = ['NN','VBD','VBP','JJ']
        pos_weights = [0.6 if term[1] in high else 0.01 if term[1] in med else 0.001 if term[1] in low else 1 for term in POS]
        custom_low = ['i', 'said', 'he', 'went','they', 'can', 'this', 'a', 'also', 'lot', 'come', 'you', 'we', 'us']
        weights = [1 if word not in custom_low else 0.1 for word in self.reduced_vocab_count.keys()]
        return [pos_weights[i] * weights[i] for i in range(len(pos_weights))]



def pairwise_distance(X):
    N = X.shape[0]
    dists = np.zeros((N, N))
    for i, a in enumerate(X):
        for j, b in enumerate(X):
            dists[i, j] = np.linalg.norm(a - b)

    return dists


def get_score(vecs):
    dists = pairwise_distance(vecs)
    mxidx = np.argmin(dists[0][1:-1]) + 1
    next_best = np.linalg.norm(vecs[mxidx] - vecs[0])
    target = np.linalg.norm(vecs[-1] - vecs[0])
    score = next_best / target
    print('SCORE: ', score)


if __name__ == '__main__':
    docs = ['People who see ghosts',
            '"I dont believe people who see ghosts", said Mannie, before spitting into the wind and riding his bike down the street at top speed. He then went home and ate peanut-butter and jelly sandwiches all day. Mannie really liked peanut-butter and jelly sandwiches. He ate them so much that his poor mother had to purchase a new jar of peanut butter every afternoon.',
            'People see incredible things. One time I saw some people talking about things they were seeing, and those people were so much fun. They saw clouds and they saw airplanes. They saw dirt and they saw worms. Can you believe the amount of seeing done by these people? People are the best.',
            'This is an article about a circus. A Circus is where people go to see other people who perform great things. Circuses also have elephants and tigers, which generally get a big woop from the crowd.',
            'Lots of people have come down with Coronavirus. You can see the latest numbers and follow our updates on the pandemic below. Please, stay safe.',
            'Goats are lovely creatures. Many people love goats. People who love goats love seeing them play in the fields.',
            'We have collected a report of people in our community seeing ghosts. Each resident was asked "how many ghosts have you seen?", "describe the last ghost you saw", and "tell us about your mother." Afterwards, we compared the ghost reports between the different individuals, and assessed whether or not they were actually seeing these apparitions.']

    my_vectorizer = Vectorizer(docs)
    ax = sns.heatmap(pairwise_distance(my_vectorizer.tfidf_mtx))
    plt.show()

    # show that rank is the same between two distances
    # do the correct normalization
    # try other weights - downweight stop words?
    print(my_vectorizer.vocab_count.keys())
    print(get_score(my_vectorizer.term_freq_mtx))
    print(get_score(my_vectorizer.tfidf_mtx))
    print(get_score(my_vectorizer.reweighted_mtx))
