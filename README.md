# NEET Performance Analyzer

## Project Overview
The NEET Performance Analyzer is a Python-based solution that helps students assess and improve their quiz performance in preparation for the NEET exam. The tool analyzes quiz data, highlights performance gaps, and provides personalized recommendations to help students strengthen weak areas.

### Key Features
- Fetch quiz data from specified endpoints (current and historical).
- Analyze performance metrics such as accuracy by topic and difficulty.
- Generate actionable recommendations for improvement.
- Predict NEET rank based on historical quiz data.
- Visualize performance insights.

## Setup Instructions

### Prerequisites
- Python 3.x
- Required Libraries: Install the necessary packages by running:
  ```bash
  pip install pandas numpy matplotlib seaborn scikit-learn requests
  ```

### Running the Project
1. Clone the repository from GitHub or download the source files.
2. Update the following endpoints in the `main` section of the code with actual API URLs:
   ```python
   current_quiz_endpoint = "https://api.example.com/current_quiz"
   historical_quiz_endpoint = "https://api.example.com/historical_quiz"
   ```
3. Run the script:
   ```bash
   python neet_performance_analyzer.py
   ```
4. View the personalized recommendations and visualizations.

## Approach Description

### Data Ingestion
- Fetches current and historical quiz data using API requests.
- Validates API responses to ensure proper data retrieval.

### Performance Analysis
- Computes topic-wise accuracy by dividing correct answers by total questions.
- Aggregates performance metrics by topic and difficulty.
- Identifies weak areas where accuracy is below 50%.

### Recommendations
- Provides targeted recommendations for topics requiring additional focus.
- Offers motivational insights for students with high accuracy.

### Rank Prediction
- Uses a linear regression model to predict the student's NEET rank based on historical quiz performance.
- Scales features and trains the model on provided data.

### Visualization
- Displays a bar chart showing accuracy across different topics. 

## Insights Summary
Below are example screenshots demonstrating key visualizations and insights:

1. **Performance Visualization:**
   ![Topic-wise Accuracy Bar Chart](path_to_screenshot_1.png)

2. **Sample Recommendations:**
   ```plaintext
   - Focus more on Physics with moderate to high difficulty questions.
   - Strengthen concepts in Organic Chemistry.
   ```

3. **Predicted NEET Rank:**
   ```plaintext
   Predicted NEET Rank: 500
   ```
SOURCE CODE:

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

class NEETPerformanceAnalyzer:
    def __init__(self, current_quiz_endpoint, historical_quiz_endpoint):
        self.current_quiz_endpoint = current_quiz_endpoint
        self.historical_quiz_endpoint = historical_quiz_endpoint

    def fetch_quiz_data(self):
        current_quiz_response = requests.get(self.current_quiz_endpoint)
        historical_quiz_response = requests.get(self.historical_quiz_endpoint)
        if current_quiz_response.status_code != 200 or historical_quiz_response.status_code != 200:
            raise ValueError("Failed to fetch data from one or both endpoints.")
        return current_quiz_response.json(), historical_quiz_response.json()

    def analyze_performance(self, historical_data):
        performance_df = pd.DataFrame(historical_data)
        if performance_df.empty or 'correct_answers' not in performance_df or 'total_questions' not in performance_df:
            raise ValueError("Historical data is incomplete or malformed.")
        performance_df['accuracy'] = (performance_df['correct_answers'] / performance_df['total_questions']).fillna(0) * 100
        topic_performance = performance_df.groupby('topic').agg({'accuracy': 'mean', 'difficulty': 'mean'}).reset_index()
        print("Performance Summary by Topic:")
        print(topic_performance)
        return topic_performance

    def generate_recommendations(self, topic_performance):
        if topic_performance.empty:
            return ["No performance data available for recommendations."]
        weak_topics = topic_performance[topic_performance['accuracy'] < 50]
        if weak_topics.empty:
            return ["Great job! Keep maintaining your current preparation."]
        return [f"Focus more on {row['topic']} with moderate to high difficulty questions." for _, row in weak_topics.iterrows()]

    def predict_neet_rank(self, historical_data):
        df = pd.DataFrame(historical_data)
        if df.empty or 'total_questions' not in df or 'correct_answers' not in df or 'neet_rank' not in df:
            raise ValueError("Historical data is incomplete or malformed for rank prediction.")
        features = df[['total_questions', 'correct_answers']]
        target = df['neet_rank']
        if len(features) < 2:
            raise ValueError("Insufficient data for rank prediction.")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)
        model = LinearRegression()
        model.fit(X_scaled, target)
        predicted_rank = model.predict([X_scaled[-1]])
        return max(1, int(predicted_rank[0]))

    def visualize_performance(self, topic_performance):
        if topic_performance.empty:
            print("No data available for visualization.")
            return
        sns.barplot(data=topic_performance, x='topic', y='accuracy')
        plt.title('Topic-wise Accuracy')
        plt.xticks(rotation=45)
        plt.show()

if __name__ == "__main__":
    current_quiz_endpoint = "https://api.example.com/current_quiz"
    historical_quiz_endpoint = "https://api.example.com/historical_quiz"
    analyzer = NEETPerformanceAnalyzer(current_quiz_endpoint, historical_quiz_endpoint)

    try:
        current_data, historical_data = analyzer.fetch_quiz_data()
        topic_performance = analyzer.analyze_performance(historical_data)
        recommendations = analyzer.generate_recommendations(topic_performance)
        print("\nPersonalized Recommendations:")
        for rec in recommendations:
            print("-", rec)
        predicted_rank = analyzer.predict_neet_rank(historical_data)
        print(f"\nPredicted NEET Rank: {predicted_rank}")
        analyzer.visualize_performance(topic_performance)
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


## Conclusion
This project provides actionable insights and recommendations for NEET preparation, empowering students with data-driven study strategies.



