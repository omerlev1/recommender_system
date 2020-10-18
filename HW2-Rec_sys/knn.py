import numpy as np
import pandas as pd
from interface import Regressor
from utils import get_data, Config
from simple_mean import SimpleMean
import pickle
from config import CORRELATION_PARAMS_FILE_PATH
import os
from tqdm import tqdm


class KnnItemSimilarity(Regressor):
    def __init__(self, config):
        self.corr_dict = {}
        self.users_means = {}
        self.items_means = {}
        self.K = config.k

    def fit(self, X: np.array):
        simple_mean_obj = SimpleMean()
        simple_mean_obj.fit(X)
        self.users_means = simple_mean_obj.user_means
        self.items_means = pd.DataFrame(X,columns=['user_index','item_index','Ratings_Rating']).groupby(['item_index'])['Ratings_Rating'].mean().reset_index().set_index('item_index').T.to_dict('list')
        if(os.path.exists(CORRELATION_PARAMS_FILE_PATH)):
            self.upload_params()
        else:
            self.build_item_to_itm_corr_dict(X)
            pickle.dump(self.corr_dict,open(CORRELATION_PARAMS_FILE_PATH,'wb'))


    def build_item_to_itm_corr_dict(self, data):
        items = np.unique(data[:,1])
        for i in tqdm(range(0,len(items))):
            perfect_corr = 0
            df = pd.DataFrame(data,columns=['user','item','rank'])
            relevant_users = df[df.item==i].loc[:,'user']
            df = df[df.user.isin(np.unique(relevant_users))]
            for j in np.unique(df[['item']]):
            # for j in range(i+1 ,len(items)):   
                if(i==j):
                    continue        
                if((j,i) in self.corr_dict):
                    continue
                item_1 = items[i]
                item_2 = items[j]
                nominator = 0
                denom_left = 0
                denom_right = 0
                df = pd.DataFrame(data,columns=['user','item','rank'])
                df_filtered = df[df['item'].isin([item_1,item_2])]
                # user_unq = np.unique(df_filtered.iloc[:,0])
                users_filter =  df_filtered['user'].value_counts()>1
                list_ranking_users = df_filtered['user'].value_counts()[users_filter].index
                df_filtered = df_filtered[df_filtered.user.isin(list_ranking_users)]
                if(df_filtered.shape[0]==0):
                    continue
                for user in list_ranking_users:
                    _df_filtered = df_filtered[df_filtered['user']==user]
                    #check if you get more than 1 line here
                    rk_item_1 =  _df_filtered[(_df_filtered.item==item_1)]
                    rk_item_2 =  _df_filtered[(_df_filtered.item==item_2)]
                    # if((rk_item_1.shape[0]==0) | (rk_item_2.shape[0]==0)):
                    #     continue
                    A = rk_item_1.iloc[0,2] - self.items_means[item_1]
                    B = rk_item_2.iloc[0,2] - self.items_means[item_2]
                    nominator += (A * B)
                    denom_left += (A**2)
                    denom_right += (B**2)
                if((denom_left==0) | (denom_right==0)):
                        continue
                sim = (nominator / np.sqrt(denom_left*denom_right))[0]
                if(sim==1):
                    perfect_corr+=1
                if(sim>0):
                    self.corr_dict.update({(item_1,item_2) : sim})
                    # print({(item_1,item_2) : sim})
                if(perfect_corr==self.K):
                    break


    def predict_on_pair(self, user, item):
        relevant_keys = [k for k,v in self.corr_dict.items() if k[0] == item]
        relevant_dict = { k: self.corr_dict[k] for k in relevant_keys }
        most_sim_items = sorted(relevant_dict.items(), key=lambda kv: kv[1],reverse=True)[:self.K]
        for item in most_sim_items.keys():
            sum = 0


    def upload_params(self):
        self.corr_dict = pickle.load(open(CORRELATION_PARAMS_FILE_PATH,'rb'))

    def save_params(self):
        raise NotImplementedError


if __name__ == '__main__':
    knn_config = Config(k=25)
    train, validation = get_data()
    knn = KnnItemSimilarity(knn_config)
    knn.fit(train)
    # print(knn.calculate_rmse(validation))
