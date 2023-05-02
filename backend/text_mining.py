from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
import preprocessing
import numpy as np
from fuzzywuzzy import process, fuzz

playlist_name_to_index = None
index_to_playlist_name = None
playlist_names_compressed = None
songs_compressed = None
song_vectorizer = None
td_matrix = None
playlist_name_vectorizer = None


def init():
    global song_vectorizer, playlist_name_to_index, index_to_playlist_name, playlist_names_compressed, songs_compressed, td_matrix, playlist_name_vectorizer
    song_vectorizer = TfidfVectorizer(
        tokenizer=lambda x: x, lowercase=False, max_df=0.7, min_df=1
    )
    song_playlist_matrix = song_vectorizer.fit_transform(
        [preprocessing.documents[x] for x in preprocessing.documents]
    )

    songs_compressed, s, playlist_names_compressed = svds(song_playlist_matrix, k=100)
    playlist_names_compressed = normalize(playlist_names_compressed.T, axis=1)
    songs_compressed = normalize(songs_compressed)

    playlist_name_to_index = song_vectorizer.vocabulary_
    index_to_playlist_name = {i: t for t, i in playlist_name_to_index.items()}

    playlist_name_vectorizer = TfidfVectorizer(
        stop_words="english", max_df=0.7, min_df=1
    )
    td_matrix = playlist_name_vectorizer.fit_transform(
        list(playlist_name_to_index.keys())
    )


def closest_playlist_names(query):
    """
    Get closest playlist names using cosine similarity.
    """
    query_vec = playlist_name_vectorizer.transform([query]).toarray()
    sims = cosine_similarity(query_vec, td_matrix)
    asort = np.argsort(-sims)
    res = [list(playlist_name_to_index.keys())[i] for i in asort[0] if sims[0][i] > 0.6]
    return res


def closest_songs_to_query(query, k=10):
    if all(isinstance(item, str) for item in query):
        # Query is playlist names - vectorize.
        query_tfidf = song_vectorizer.transform([query]).toarray()
        query_vec = normalize(np.dot(query_tfidf, playlist_names_compressed)).squeeze()
    elif all(isinstance(item, str) for item in query):
        # Query is already vectorized (from rocchio)
        query_vec = query
    sims = songs_compressed.dot(query_vec)
    asort = np.argsort(-sims)[:k]
    return [(list(preprocessing.documents.items())[i][0], sims[i]) for i in asort]


def rocchio(q0, relevant_songs, irrelevant_songs, a=0.3, b=0.3, c=0.8, clip=True):
    q1 = a * q0
    if len(relevant_songs) > 0:
        rel_doc_sum = np.zeros((len(q0),))
        for rel_song in relevant_songs:
            rel_doc_sum += td_matrix[preprocessing.documents.keys().index(rel_song)]
        q1 += b * (1 / len(relevant_songs)) * rel_doc_sum

    if len(irrelevant_songs) > 0:
        irrel_doc_sum = np.zeros((len(q0),))
        for irrel_song in irrelevant_songs:
            irrel_doc_sum += td_matrix[preprocessing.documents.keys().index(irrel_song)]
        q1 -= c * (1 / len(irrelevant_songs)) * irrel_doc_sum

    if clip:
        q1 = np.clip(q1, 0, None)

    return q1, closest_songs_to_query(q1)
