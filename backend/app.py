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

app = Flask(__name__)
CORS(app)

preprocessing.preprocess()
text_mining.init()

curr_query = None


@app.route("/search")
def search():
    global curr_query
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
    print(query)
    print(top_songs)
    curr_query = text_mining.query_to_vec(query)
    return [song for song, score in top_songs]


@app.route("/rocchio", methods=["POST"])
def rocchio():
    global curr_query
    data = request.json
    rel_track_list = data["rel_track_list"]
    irrel_track_list = data["irrel_track_list"]
    rel_track_list = [tuple(x) for x in rel_track_list]
    irrel_track_list = [tuple(x) for x in irrel_track_list]
    k = 13
    # if both are empty then just send the results as usual
    print("Relevant:", rel_track_list)
    print("Irrelevant:", irrel_track_list)
    q0 = curr_query
    q1, top_songs = text_mining.rocchio(
        q0, rel_track_list, irrel_track_list, clip=False
    )
    results = [song for song, score in top_songs[:k]]
    print(q1)
    print(results)
    curr_query = q1
    return results
