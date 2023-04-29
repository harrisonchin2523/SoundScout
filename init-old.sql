CREATE DATABASE IF NOT EXISTS playlistsdb;

USE playlistsdb;
DROP TABLE IF EXISTS playlists;

CREATE TABLE playlists(
    user_id varchar(64),
    artistname varchar(64),
    trackname varchar(64),
    playlistname varchar(64)
);

LOAD DATA LOCAL infile '../data/spotify_dataset.csv' INTO TABLE playlists
fields terminated by ','
enclosed by '"'
lines terminated by '\n'
(user_id, artistname, trackname, playlistname)