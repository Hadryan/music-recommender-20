import random

from .collaborative_filtering_model import get_cf_recommendations, predictions_dict, predictions
from .content_based_model import get_cb_recommendations

from surprise.prediction_algorithms.predictions import Prediction

NUMBER_OF_RECOMMENDATIONS = 10

class HybridCBCFModel:
    '''
        Class that implements hybrid version of content based and collaborative filtering.
    '''

    def __init__(self, cb_model, scaler, user_database, song_database, features=['hotttness','familiarity','duration','loudness','tempo','key','mode','time_signature'], history_threshold=50):
        self.cb_model = cb_model
        self.scaler = scaler
        self.user_database = user_database
        self.song_database = song_database
        self.features = features
        self.history_threshold = history_threshold

    def get_user_history_by_id(self, user_id):
        '''
            Calculates how many 'ratings' (user plays) this particular user has.
        '''
        return self.user_database[self.user_database['userID'] == user_id].shape[0]

    def get_recommendations(self, user, recs_portion, mode='cf'):
        out_predictions = []
        mean_number_of_plays = self.user_database['playCount'].mean()
        for iid in recs_portion:
            song_already_played = self.user_database[(self.user_database['userID'] == user) & (self.user_database['songID'] == iid)]
            if song_already_played.shape[0] == 0:
                actual_plays = 0
            else:
                actual_plays = song_already_played['playCount'].iloc[0]
            estimated_rating = predictions_dict[(user,iid)] if mode == 'cf' else mean_number_of_plays
            out_predictions.append(Prediction(uid=user, iid=iid, r_ui=actual_plays, est=estimated_rating, details={}))

        return out_predictions


    def run(self):
        out_predictions = []
        for user in list(set(self.user_database['userID'])):
            cf_recs = get_cf_recommendations(user, n=NUMBER_OF_RECOMMENDATIONS)
            cb_recs = get_cb_recommendations(user_id=user, model=self.cb_model, scaler=self.scaler, user_database=self.user_database, song_database=self.song_database, features=self.features, n=NUMBER_OF_RECOMMENDATIONS)
            history = self.get_user_history_by_id(user)

            if not cf_recs or not cb_recs:
                continue

            if history > self.history_threshold:
                # rich enough history to avoid cold-start problem for CF
                out_predictions.extend(self.get_recommendations(user, cf_recs))
            else:
                # use cb approach as a help
                cf_sample_size = int(history / 5) if int(history / 5) < len(cf_recs) else len(cf_recs)
                cb_sample_size = 10 - cf_sample_size if (10 - cf_sample_size) < len(cb_recs) else len(cb_recs)

                out_predictions.extend(self.get_recommendations(user, random.sample(cf_recs, cf_sample_size)))
                out_predictions.extend(self.get_recommendations(user, random.sample(cb_recs, cb_sample_size), mode='cbm'))

        return out_predictions






