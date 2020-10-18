from interface import Regressor
from utils import get_data
import pandas as pd

class SimpleMean(Regressor):
    def __init__(self):
        self.user_means = {}

    def fit(self, X):
        self.user_means = pd.DataFrame(X,columns=['user_index','item_index','Ratings_Rating']).groupby(['user_index'])['Ratings_Rating'].mean().reset_index().set_index('user_index').T.to_dict('list')

    def predict_on_pair(self, user: int, item: int):
        return self.user_means[user]


if __name__ == '__main__':
    train, validation = get_data()
    baseline_model = SimpleMean()
    baseline_model.fit(train)
    print(baseline_model.calculate_rmse(validation))
