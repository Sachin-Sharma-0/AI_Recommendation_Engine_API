# AI-Powered Recommendation Engine for E-Commerce

This project implements an AI-powered recommendation engine to provide personalized product recommendations for an e-commerce platform. It includes data preprocessing, collaborative filtering model training, an API to fetch recommendations, and an evaluation mechanism.

## Table of Contents
- [Project Overview](#project-overview)
- [Data Collection and Preprocessing](#data-collection-and-preprocessing)
- [Model Development](#model-development)
- [Scalability and Performance](#scalability-and-performance)
- [Evaluation and Optimization](#evaluation-and-optimization)
- [API Endpoints](#api-endpoints)
- [Setup Instructions](#setup-instructions)
- [Evaluation](#evaluation)

## Project Overview
This project implements a recommendation engine using collaborative filtering to provide personalized product recommendations based on user behavior.

## Data Collection and Preprocessing
Data is collected from open-source datasets and preprocessed to remove missing values and normalize ratings.

## Model Development
A collaborative filtering model using Singular Value Decomposition (SVD) is implemented to generate recommendations.

## Scalability and Performance
The recommendation engine is designed to handle high traffic and provide low-latency responses.

## Evaluation and Optimization
The model is evaluated using RMSE and MAE metrics and optimized based on continuous feedback.

## API Endpoints
### Type: GET 
### Endpoint: /recommend/\<int:user_id\>
Returns the top 10 product recommendations for the given user ID.

## Setup Instructions
1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/AI_Recommendation_Engine_API.git
   cd AI_Recommendation_Engine_API

2. **Install Dependencies**
   ```bash
   pip install numpy==1.22.4
   pip install pandas==1.3.3
   pip install scikit-learn==0.24.2
   pip install Flask==2.0.1
   pip install surprise==0.1

3. **Run the Flask Application**
   ```bash
   python flaskApi.py

4. **Access the API endpoint:**
   ```bash
   http://127.0.0.1:5000/recommendations/<user_id>

## Project Structure
- `Dataset.py`: Script for data collection and preprocessing.
- `filtering.py`: Script for collaborative filtering model development.
- `flaskApi.py`: Script for setting up the Flask API.
- `evaluation.py`: Script for evaluating the recommendation model.

## Evaluation
Evaluate the model performance using `evaluation.py`:
```bash
python evaluation.py




