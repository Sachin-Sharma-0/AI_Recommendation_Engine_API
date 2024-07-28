from surprise import Dataset, Reader
import pandas as pd
import zipfile
import requests
import io

def download_and_preprocess():
    # Download the dataset
    url = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
    response = requests.get(url)
    zip_file = zipfile.ZipFile(io.BytesIO(response.content))
    zip_file.extractall()

    # Load the data
    data = pd.read_csv('ml-100k/u.data', sep='\t', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])

    # Preprocess the data
    data = data[['user_id', 'item_id', 'rating']]
    data['user_id'] = data['user_id'].astype(int)
    data['item_id'] = data['item_id'].astype(int)
    data['rating'] = data['rating'].astype(float)

    return data


if __name__ == "__main__":
    data = download_and_preprocess()
    print(data.head())
    data.to_csv('preprocessed_data.csv', index=False)
    