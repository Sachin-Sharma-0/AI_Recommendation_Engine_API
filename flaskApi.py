from flask import Flask, jsonify, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the preprocessed data
data = pd.read_csv('preprocessed_data.csv')

# Load the trained model
with open('svd_model.pkl', 'rb') as f:
    algo = pickle.load(f)

@app.route('/recommendations/<int:user_id>', methods=['GET'])
def get_recommendations(user_id):
    # Check if the user_id exists in the dataset
    if user_id not in data['user_id'].unique():
        return jsonify({"error": "User ID not found"}), 404

    # Generate predictions for the user
    user_data = data[data['user_id'] == user_id]
    recommendations = []

    for item_id in data['item_id'].unique():
        prediction = algo.predict(user_id, item_id)
        recommendations.append((item_id, prediction.est))

    # Sort recommendations by estimated rating
    recommendations.sort(key=lambda x: x[1], reverse=True)
    
    # Convert to a readable format with int conversion
    top_recommendations = [{"item_id": int(item[0]), "predicted_rating": item[1]} for item in recommendations[:10]]

    return jsonify(top_recommendations)


if __name__ == '__main__':
    app.run(debug=True)
