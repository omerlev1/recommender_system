import pandas as pd
import numpy as np

def get_data():
    """
    reads train, validation to python indices so we don't need to deal with it in each algorithm.
    of course, we 'learn' the indices (a mapping from the old indices to the new ones) only on the train set.
    if in the validation set there is an index that does not appear in the train set then we can put np.nan or
     other indicator that tells us that.
    """
    # return train, validation
    train = pd.read_csv('C:/Users/omer.l/Documents/HW2-Rec_sys/data/Train.csv')
    validation = pd.read_csv('C:/Users/omer.l/Documents/HW2-Rec_sys/data/Validation.csv')
    unq_user_id = train['User_ID_Alias'].unique()
    unq_movie_id = train['Movie_ID_Alias'].unique()
    
    users_dict = {}
    idx=0
    for user in unq_user_id:
        if(user in users_dict):
            continue
        users_dict.update({user:idx})
        idx+=1
    
    item_dict = {}
    idx=0
    for item in unq_movie_id:
        if(item in item_dict):
            continue
        item_dict.update({item:idx})
        idx+=1

    df_user_idx = pd.DataFrame({'user':list(users_dict.keys()),'user_index':list(users_dict.values())})
    df_item_idx = pd.DataFrame({'item':list(item_dict.keys()),'item_index':list(item_dict.values())})
    #train
    df_with_idx = pd.merge(left = train, right = df_user_idx, left_on = 'User_ID_Alias',right_on = 'user',how='left').drop(['user'],axis=1)
    df_with_idx = pd.merge(left = df_with_idx, right = df_item_idx, left_on = 'Movie_ID_Alias',right_on = 'item',how='left').drop(['item'],axis=1)
    #tvalidation
    df_with_idx_v = pd.merge(left = validation, right = df_user_idx, left_on = 'User_ID_Alias',right_on = 'user',how='left').drop(['user'],axis=1)
    df_with_idx_v = pd.merge(left = df_with_idx_v, right = df_item_idx, left_on = 'Movie_ID_Alias',right_on = 'item',how='left').drop(['item'],axis=1)
    train = df_with_idx[['user_index','item_index','Ratings_Rating']].to_numpy()
    validation = df_with_idx[['user_index','item_index','Ratings_Rating']].to_numpy()
    return train, validation


class Config:
    def __init__(self, **kwargs):
        self._set_attributes(kwargs)

    def _set_attributes(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
