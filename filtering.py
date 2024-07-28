from surprise import Dataset, Reader
from surprise import SVD
from surprise import accuracy
from surprise.model_selection import train_test_split
import pandas as pd
import pickle

def train_model(data):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(data[['user_id', 'item_id', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.25)
    
    algo = SVD()
    algo.fit(trainset)
    predictions = algo.test(testset)
    
    rmse = accuracy.rmse(predictions)
    print(f'RMSE: {rmse}')
    
    with open('svd_model.pkl', 'wb') as f:
        pickle.dump(algo, f)

if __name__ == "__main__":
    data = pd.read_csv('preprocessed_data.csv')
    train_model(data)