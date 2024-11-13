# Health Risk Prediction with tkinter

This application predicts health risks based on environmental factors for a specified city using historical environmental data. It fetches data via APIs, processes it, and provides a risk-level color-coded display.

## Features

**Geolocation & Data Fetching**: Automatically fetches latitude and longitude for a city and retrieves historical environmental data (e.g., PM2.5, AQI, temperature, humidity).

**Risk Prediction**: Displays color-coded risk levels for air quality and environmental parameters.

**Interactive Map**: Opens the city location in Google Maps.

**GUI**: User-friendly interface built using Tkinter.

## Setup

### Clone this repository:

_git clone <repository-url>_

_cd <repository-directory>_

Install the required Python libraries:

_pip install -r requirements.txt_
_Insert your API keys for OpenWeather and Visual Crossing in the code._

## Usage
### 1. Run the application:

_python Health_Risk_Prediction_with_tkinter.py_

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
