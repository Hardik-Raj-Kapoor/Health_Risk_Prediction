# Health Risk Prediction with tkinter

This application predicts health risks based on environmental factors for a specified city using historical environmental data. It fetches data via APIs, processes it, and provides a risk-level color-coded display.

## Features

**Geolocation & Data Fetching**: Automatically fetches latitude and longitude for a city and retrieves historical environmental data (e.g., PM2.5, AQI, temperature, humidity).

**Risk Prediction**: Displays color-coded risk levels for air quality and environmental parameters.

**Interactive Map**: Opens the city location in Google Maps.

**GUI**: User-friendly interface built using Tkinter.

## Setup

### Clone this repository:
```
git clone <repository-url>

cd <repository-directory>
```
Install the required Python libraries:
```
pip install -r requirements.txt
```
_Insert your API keys for OpenWeather and Visual Crossing in the code._

## Usage
### 1. Run the application:
```
python Health_Risk_Prediction_with_tkinter.py
```
### 2. Enter the desired city in the app's text box.


### 3. Click Predict Health Risk. This will:

  + Fetch environmental data and display it in a color-coded table.
  + Open Google Maps to the specified city.

## Code Overview
+ `fetch_geo_location(city):` Retrieves latitude and longitude.
+ `fetch_environmental_data(lat, lon, date):` Fetches historical environmental data.
+ `predict_health_risk():` Gathers data and initiates risk-level display.
+ `display_environmental_data(data):` Displays data in a color-coded table for risk analysis.

## Requirements
+ Python 3.7+
+ Libraries: `pandas`, `numpy`, `requests`, `scikit-learn`, `tensorflow`, `tkinter`, `matplotlib`, `seaborn`
+ API Keys for OpenWeather and Visual Crossing

##  Example
Upon entering **"New York"** and clicking **Predict Health Risk**, you will see a Google Maps window showing New York's location and a table with 30-day environmental data, with color-coded risk levels.
