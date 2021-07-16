import matplotlib.pyplot as plt
import pandas as pd
from surprise import Dataset, Reader
from surprise import SVD
from surprise.model_selection import train_test_split
from collections import defaultdict

user_data = pd.read_csv('./output_plays_small_preprocessed.csv')
user_data.columns = ['userID', 'songID', 'playCount']

song_df = pd.read_csv('./output.csv')
dataset = Dataset.load_from_df(user_data, Reader(rating_scale=(1, user_data['playCount'].max())))

print("Building trainset and testset...")
trainset, testset = train_test_split(dataset, test_size=0.20)
svd = SVD()

print("Fitting on trainset...")
svd.fit(trainset)

print("Creating predictions on test set...")
predictions = svd.test(testset)

predictions_dict = {}
for uid, iid, _, est, _ in predictions:
    predictions_dict[(uid,iid)] = est

def get_cf_recommendations(user_id, n=10):
    '''
        Function that returns top n recommendations.
    '''
    # get predictions for specific user
    user_predictions = [prediction for prediction in predictions if prediction[0] == user_id]

    if not user_predictions:
        return []

    # sort predictions in descending order of estimation score
    user_predictions.sort(key=lambda el: el[3], reverse=True)

    # take n predictions
    user_predictions = user_predictions[:n]
    
    recommendations = []
    for user_id, song_id, _, est, _ in user_predictions:
        # if song_df[song_df['song_id'] == song_id].empty:
        #     continue
        # song_title = song_df[song_df['song_id'] == song_id]['title'].to_string().split('    ')[1]
        # artist = song_df[song_df['song_id'] == song_id]['artist_name'].to_string().split('    ')[1]
        # year = song_df[song_df['song_id'] == song_id]['year'].to_string().split('    ')[1]
        # recommendations.append({'title': song_title, 'artist': artist, 'year': year})
        recommendations.append(song_id)
    
    return recommendations
