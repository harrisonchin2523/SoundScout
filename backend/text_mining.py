from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from scipy.sparse.linalg import svds
import preprocessing
import numpy as np
from fuzzywuzzy import process, fuzz

song_to_index = None
index_to_song = None
songs_compressed = None
docs_compressed = None


def init():
    global song_to_index, index_to_song, songs_compressed, docs_compressed
    vectorizer = TfidfVectorizer(
        tokenizer=lambda x: x, lowercase=False, max_df=0.7, min_df=1
    )
    td_matrix = vectorizer.fit_transform(
        [preprocessing.documents[x] for x in preprocessing.documents]
    )

    docs_compressed, s, songs_compressed = svds(td_matrix, k=40)
    songs_compressed = normalize(songs_compressed.T, axis=1)
    docs_compressed = normalize(docs_compressed)

    song_to_index = vectorizer.vocabulary_
    index_to_song = {i: t for t, i in song_to_index.items()}


def top_songs(closest_playlists, k=15):
    song_scores = {}
    for playlist_name, tracks, similarity in closest_playlists:
        for track in tracks:
            if track not in song_scores:
                song_scores[track] = 0
            song_scores[track] += similarity

    ranked_songs = list(song_scores.items())
    ranked_songs.sort(key=lambda x: x[1], reverse=True)
    return ranked_songs[:k]


def closest_playlists(playlist_name, k=10):
    index = list(preprocessing.documents.keys()).index(playlist_name)
    sims = docs_compressed.dot(docs_compressed[index, :])
    asort = np.argsort(-sims)[:k]

    res = []
    num_playlists = 0
    for i in asort:
        if num_playlists < k:
            name, tracks = list(preprocessing.documents.items())[i]
            num_playlists += preprocessing.title_histogram[name]
            res.append((name, tracks, sims[i]))
        else:
            break

    return res
