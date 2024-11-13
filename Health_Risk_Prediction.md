# Health_Risk_Prediction
This project aims to predict health risk indices based on environmental data (e.g., air quality, temperature, humidity) for a given city. Using the model’s output, the code classifies health risks (e.g., low, moderate, high, or critical) and provides health recommendations along with probabilities for various diseases. Additionally, it forecasts health risk indices for the next few days based on future environmental data.

## Features
Fetch Environmental Data: Retrieves historical and future environmental data (e.g., PM2.5, PM10, temperature, humidity, UV index) for a city from the Visual Crossing API and geolocation data from the OpenWeatherMap API.

Health Risk Index Prediction: Generates a health risk index based on environmental factors.

Classification and Recommendations: Provides a risk classification (low, moderate, high, critical) and personalized health recommendations.

Disease Probabilities: Outputs the probabilities of potential diseases related to air pollution and other environmental factors.

Visualization: Plots historical and predicted health risk indices and environmental trends over time.

## Technologies Used
Python Libraries: pandas, numpy, requests, matplotlib, seaborn, scikit-learn, tensorflow

APIs: Visual Crossing API, OpenWeatherMap API

Machine Learning Model: Deep Neural Network using TensorFlow Keras 

## Setup
### Clone the repository:

*git clone [https://github.com/Hardik-Raj-Kapoor/health-risk-prediction.git](https://github.com/Hardik-Raj-Kapoor/Health_Risk_Prediction)*

*cd health-risk-prediction*

### Install Dependencies:

*pip install -r requirements.txt*

### API Keys:

Register for API keys at Visual Crossing and OpenWeatherMap.

Replace placeholders with your keys in the code:

*visual_crossing_api_key = 'YOUR_VISUAL_CROSSING_API_KEY'*

*openweather_api_key = 'YOUR_OPENWEATHERMAP_API_KEY'*

## Usage
### Run the Script:

*python health_risk_prediction.py*

*Input City Name: You’ll be prompted to enter a city name.*

### Output: The script will display:

+ Environmental data for the past month
+ Correlation heatmap of environmental factors
+ Health risk classification and recommended actions
+ Disease probability estimates
+ Predicted health risk indices for the next 4 days
+ Visualization of historical and forecasted environmental factors and health risks

## Project Structure
+ health_risk_prediction.py - The main script for data fetching, processing, model training, and predictions.
+ requirements.txt - Lists required Python libraries.
+ API keys (not included in the repository) - Register and add your own API keys for functionality.

## Future Scope
This project can be enhanced in several ways:

+ Extended Model Training: Include additional environmental and weather data for more accurate predictions.

+ User-Friendly Dashboard: Implement a web or mobile application interface for accessibility.

+ Additional Risk Factors: Integrate other health risk factors like age, pre-existing health conditions, etc., for personalized predictions.


