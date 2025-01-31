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

## Conclusion
This project provides actionable insights and recommendations for NEET preparation, empowering students with data-driven study strategies.



