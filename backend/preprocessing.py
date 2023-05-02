import os
import json
import nltk
from nltk.corpus import stopwords
import numpy as np
import math
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.linalg import svds
from unidecode import unidecode
import collections

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
documents = {}  # song : [playlist1, playlist2, ...]
title_histogram = collections.Counter()


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
                process_playlist(playlist)


def process_playlist(playlist):
    global total_playlists, total_tracks, title_histogram
    total_playlists += 1
    nname = normalize_name(playlist["name"])

    # Create documents for SVD
    for track in playlist["tracks"]:
        total_tracks += 1
        song = (track["track_name"], track["artist"], track["uri"])
        if song not in documents:
            documents[song] = []
        documents[song].append(nname)

    title_histogram[nname] += 1


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
    tokens = [unidecode(tok) for tok in tokens if tok not in stopWords]
    return tokens


def preprocess():
    print("Processing...")
    process_mpd("data")
    print("Total playlists:", total_playlists)
    print("Done")
