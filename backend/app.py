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

# ROOT_PATH for linking with all your files.
# Feel free to use a config.py or settings.py with a global export variable
# os.environ["ROOT_PATH"] = os.path.abspath(os.path.join("..", os.curdir))

# # These are the DB credentials for your OWN MySQL
# # Don't worry about the deployment credentials, those are fixed
# # You can use a different DB name if you want to
# MYSQL_USER = "root"
# MYSQL_USER_PASSWORD = os.getenv("MYSQL_USER_PASSWORD")
# MYSQL_PORT = 3306
# MYSQL_DATABASE = "playlistsdb"

# mysql_engine = MySQLDatabaseHandler(
#     MYSQL_USER, MYSQL_USER_PASSWORD, MYSQL_PORT, MYSQL_DATABASE
# )

# # Path to init.sql file. This file can be replaced with your own file for testing on localhost, but do NOT move the init.sql file
# # albert xiao is so hot
# mysql_engine.load_file_into_db()

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
