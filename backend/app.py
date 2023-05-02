import json
import os
import re
import math
import numpy as np
import nltk
from nltk.corpus import stopwords
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
from dotenv import load_dotenv
import preprocessing
import text_mining
import ssl
from fuzzywuzzy import process, fuzz

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


load_dotenv()
nltk.download("wordnet")
nltk.download("punkt")

# albert xiao is so hot
app = Flask(__name__)
CORS(app)

preprocessing.preprocess()
text_mining.init()


@app.route("/search")
def search():
    query = request.args.get("title")
    query = preprocessing.normalize_name(query)
    top_songs = []
    k = 14
    if query not in list(text_mining.playlist_name_to_index.keys()):
        print("Not an existing playlist name.")
        # Find best matches to existing playlist names
        query = " ".join(preprocessing.tokenize(query))
        closest_playlist_names = text_mining.closest_playlist_names(query)
        print("Matching with:", closest_playlist_names)
        if len(closest_playlist_names) > 0:
            # Query terms exist in playlist titles
            top_songs = text_mining.closest_songs_to_query(closest_playlist_names, k=k)
            query = closest_playlist_names
        else:
            # Find closest literal playlist title
            closest_name, score = process.extract(
                query,
                list(text_mining.playlist_name_to_index.keys()),
                scorer=fuzz.token_set_ratio,
            )[0]
            top_songs = text_mining.closest_songs_to_query([closest_name], k=k)
            query = [closest_name]
    else:
        # Query is existing playlist name
        top_songs = text_mining.closest_songs_to_query([query], k=k)
        query = [query]
    print(top_songs)
    return query, [song for song, score in top_songs]


@app.route("/rocchio", methods=["POST"])
def rocchio():
    data = request.json
    rel_track_list = data["rel_track_list"]
    irrel_track_list = data["irrel_track_list"]
    rel_track_list = [tuple(x) for x in rel_track_list]
    irrel_track_list = [tuple(x) for x in irrel_track_list]
    k = 15
    # if both are empty then just send the results as usual
    print("Relevant:", rel_track_list)
    print("Irrelevant:", irrel_track_list)
    q1, top_songs = text_mining.rocchio(
        [], rel_track_list, irrel_track_list, clip=False
    )
    results = [song for song, score in top_songs[:k]]
    print(results)
    return q1, results
