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
    k = 15
    if query not in list(preprocessing.documents.keys()):
        print("Not an existing playlist name.")
        # Find best matches to existing playlist names
        query = " ".join(preprocessing.tokenize(query))
        closest_playlist_names = process.extract(
            query,
            list(preprocessing.documents.keys()),
            scorer=fuzz.token_set_ratio,
        )
        print(closest_playlist_names)
        if closest_playlist_names[0][1] == 100:
            exact_matches = [
                name for (name, score) in closest_playlist_names if score == 100
            ]
            for match in exact_matches:
                # Get best songs of match
                closest_playlists = text_mining.closest_playlists(
                    match, k=10 // len(exact_matches)
                )
                for song_info, score in text_mining.top_songs(closest_playlists)[
                    : k // len(exact_matches)
                ]:
                    top_songs.append((song_info, score))
            print(top_songs[:k])
            top_songs.sort(key=lambda x: x[1], reverse=True)
            return [song for song, score in top_songs[:k]]
        else:
            # No optimal matches, settle on closest existing name
            query = closest_playlist_names[0][0]
    # Query is existing playlist name
    closest_playlists = text_mining.closest_playlists(query, k=10)
    for name, _, _ in closest_playlists:
        print(name)
    top_songs.extend(text_mining.top_songs(closest_playlists))
    print(top_songs[:k])
    top_songs.sort(key=lambda x: x[1], reverse=True)
    return [song for song, score in top_songs[:k]]


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
    closest_playlists = text_mining.regenerate_closest_playlists(
        rel_track_list, irrel_track_list
    )
    top_songs = text_mining.top_songs(closest_playlists)
    top_songs.sort(key=lambda x: x[1], reverse=True)
    results = [song for song, score in top_songs[:k]]
    print(results)
    return results
