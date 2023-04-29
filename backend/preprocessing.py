import os
import json
import nltk
from nltk.corpus import stopwords
import numpy as np
import math
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.linalg import svds

stopWords = set(stopwords.words("english"))
stopWords = stopWords.union(
    {
        "song",
        "playlist",
        "music",
    }
)

total_playlists = 0
total_tracks = 0
idf = None
inv_idx = None  # term: [(pid, tf), ...]
playlists = {}  # pid: playlist JSON
descriptions = {}  # name: [desc1, desc2, ...]
doc_norms = None
documents = []


def process_mpd(path):
    global inv_idx
    filenames = os.listdir(path)
    for filename in sorted(filenames):
        if filename.endswith(".json"):
            fullpath = os.sep.join((path, filename))
            f = open(fullpath)
            js = f.read()
            f.close()
            mpd_slice = json.loads(js)

            for playlist in mpd_slice["playlists"]:
                playlists[playlist["pid"]] = playlist
                process_playlist(playlist)

            description_tfs = process_descriptions()
            inv_idx = create_inverted_index(mpd_slice["playlists"], description_tfs)


def process_playlist(playlist):
    global total_playlists, total_tracks
    total_playlists += 1

    # Create documents for SVD
    playlist_tracks = []
    for track in playlist["tracks"]:
        total_tracks += 1
        playlist_tracks.append(track["track_name"])

    nname = normalize_name(playlist["name"])
    documents.append((nname, playlist_tracks))

    # Update descriptions dict
    if "description" in playlist:
        desc = playlist["description"]
        ndesc = normalize_name(desc)
        if nname not in descriptions:
            descriptions[nname] = []
        descriptions[nname].append(ndesc)


def create_inverted_index(playlists, description_tfs):
    inv_idx = {}
    for playlist in playlists:
        nname = normalize_name(playlist["name"])
        tokens = tokenize(nname)
        # Get tfs for terms in playlist name
        tfs = nltk.FreqDist(tokens)

        # Add tfs for terms in playlist description
        if nname in description_tfs:
            tfs.update(description_tfs[nname])

        # Update inverted index
        for tok in tfs:
            if tok not in inv_idx:
                inv_idx[tok] = []
            inv_idx[tok].append((playlist["pid"], tfs[tok]))

    return inv_idx


def compute_idf(inv_idx, n_docs, min_df=2, max_df_ratio=0.2):
    idf = {}
    for term in inv_idx:
        df = len(inv_idx[term])
        if min_df <= df <= max_df_ratio * n_docs:
            idf[term] = math.log2(n_docs / (1 + df))
    return idf


def compute_doc_norms(inv_idx, idf, n_docs):
    """
    Precompute the euclidean norm of each document.
    """
    norms = np.zeros((n_docs,))
    for i in inv_idx:
        if i in idf:
            for j, tf_ij in inv_idx[i]:
                norms[j] += (tf_ij * idf[i]) ** 2

    norms = np.sqrt(norms)
    return norms


def normalize_name(name):
    """
    Normalizes a string by converting it to lowercase, removing special characters and extra spaces, and returning the result.
    """
    name = name.lower()
    name = re.sub(r"[.,\/#!$%\^\*;:{}=\_`~()@]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def tokenize(s, lemmatizer=nltk.WordNetLemmatizer()):
    s = normalize_name(s)
    tokens = nltk.word_tokenize(s)
    tokens = [lemmatizer.lemmatize(tok) for tok in tokens]
    tokens = [tok for tok in tokens if tok not in stopWords]
    return tokens


def process_descriptions():
    description_tfs = {}
    for nname, ndescs in descriptions.items():
        if nname in description_tfs:
            continue
        num_descs = len(ndescs)
        combined_desc = " ".join(ndescs)
        tokens = tokenize(combined_desc)
        tfs = nltk.FreqDist(tokens)

        # Average term frequencies of descriptions
        tfs = {term: tfs[term] / num_descs for term in tfs}
        description_tfs[nname] = tfs

    return description_tfs


def preprocess():
    global idf, inv_idx, doc_norms
    print("Processing...")
    process_mpd("data")
    print("Total playlists:", total_playlists)

    print("Computing idf...")
    idf = compute_idf(inv_idx, total_playlists)
    inv_idx = {key: val for key, val in inv_idx.items() if key in idf}  # Prune inv_idx

    print("Computing doc norms...")
    doc_norms = compute_doc_norms(inv_idx, idf, total_playlists)

    print("Done")
