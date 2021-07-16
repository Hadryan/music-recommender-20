import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import sklearn.preprocessing as preprocessing


def build_song_database(filepath='./output.csv', features=['hotttness','familiarity','duration','loudness','tempo','key','mode','time_signature'], scaler=preprocessing.RobustScaler()):
    '''
        Function that builds database with song specific feature data.
    '''
    # load data
    song_data_df = pd.read_csv(filepath, header=0)

    # duplicate removal
    song_data_df = song_data_df.drop_duplicates()

    song_data_df = song_data_df.reset_index(drop=True)

    train_data = song_data_df[features]

    train_data_normalized = scaler.fit_transform(train_data)

    return train_data_normalized, song_data_df, features, scaler


def build_user_database(filepath='./output_plays.csv'):
    '''
        Function that builds database with user, song, number of plays triplets.
    '''
    user_data = pd.read_csv(filepath)
    user_data.columns = ['userID', 'songID', 'playCount']
    return user_data



def build_model(train_data_normalized, k=1000, dist_metric='manhattan'):
    model = NearestNeighbors(k, metric=dist_metric)
    model.fit(train_data_normalized)
    return model


def get_cb_recommendations(user_id, model, scaler, user_database, song_database, features, n=100):
    # get all songs for the user
    user_i_data = user_database.loc[user_database['userID'] == user_id]

    # check if user has any songs played
    if len(user_i_data) == 0:
        raise Exception(f'No songs for user with id {user_id}')

    # create a list of all the songs that this user listens to
    user_i_songs = user_i_data['songID'].unique().tolist()

    total_playcount = user_i_data['playCount'].sum()
    total_playcount = 1 if not total_playcount else total_playcount

    recommended_songs = []

    for song in user_i_songs:
        song_data = song_database.loc[song_database['song_id']==song]
        song_data = song_data[features]
        if song_data.empty:
            continue

        scaled_song_data = scaler.transform(song_data)
        similar_song_ids = model.kneighbors(scaled_song_data, return_distance=False).flatten()

        s_playcount = int(user_i_data.loc[user_i_data['songID']==song,'playCount'])

        top_songs = similar_song_ids[1:max(int(1 + round(n*s_playcount/total_playcount)),2)]

        recommended_songs.extend(song_database.iloc[top_songs,:][['song_id']]['song_id'].tolist())


    if len(recommended_songs) > n:
        recommended_songs = np.random.choice(recommended_songs, n, replace=False).tolist()

    return recommended_songs