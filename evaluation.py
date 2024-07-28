from surprise import Dataset, Reader
from surprise import accuracy
from surprise.model_selection import train_test_split
import pandas as pd
import pickle

def evaluate_model(data):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(data[['user_id', 'item_id', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.25)
    
    # Load the trained model
    with open('svd_model.pkl', 'rb') as f:
        algo = pickle.load(f)
    
    predictions = algo.test(testset)
    
    # Compute RMSE
    rmse = accuracy.rmse(predictions)
    print(f'RMSE: {rmse}')

if __name__ == "__main__":
    data = pd.read_csv('preprocessed_data.csv')
    evaluate_model(data)

