from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from scipy.sparse.linalg import svds
import preprocessing
import numpy as np

word_to_index = None
index_to_word = None
songs_compressed = None


def init():
    global word_to_index, index_to_word, songs_compressed
    vectorizer = TfidfVectorizer(
        tokenizer=lambda x: x, lowercase=False, max_df=0.7, min_df=5
    )
    td_matrix = vectorizer.fit_transform([x[1] for x in preprocessing.documents])

    docs_compressed, s, songs_compressed = svds(td_matrix, k=40)
    songs_compressed = normalize(songs_compressed.T, axis=1)

    word_to_index = vectorizer.vocabulary_
    index_to_word = {i: t for t, i in word_to_index.items()}


# cosine similarity
def closest_songs(song_in, k=15):
    if song_in not in word_to_index:
        raise Exception("Not in dataset")
    sims = songs_compressed.dot(songs_compressed[word_to_index[song_in], :])
    asort = np.argsort(-sims)[: k + 1]
    return [(index_to_word[i], sims[i]) for i in asort[1:]]
