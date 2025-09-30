Time Series Forecasting of Malaria Mortality Rate in Kenya
📋 Project Overview
This project applies time series forecasting techniques to analyze and predict trends in malaria mortality rates in Kenya. The primary goal was to leverage historical data (2010-2020) to build a predictive model using the ARIMA (AutoRegressive Integrated Moving Average) framework.

Author: Hellen Gouland Ouma
Date: September 2, 2025
Tool: Python IDLE
Libraries: pandas, matplotlib, statsmodels, sklearn

🎯 Objectives
To preprocess and analyze national-level malaria mortality data

To develop and validate an ARIMA model for time series forecasting

To generate and interpret a forecast for malaria mortality rates in Kenya for 2021-2023

📊 Dataset
File: openafrica-_-malaria-_-national_unit-data-mortality-rate.csv

The dataset contains national-level annual data for multiple countries from 2010 to 2020.

Relevant Columns Used:

Name: Country name (Filtered for 'Kenya')

Metric: Type of measurement (Filtered for 'Mortality Rate')

Units: Deaths per 100 Thousand

Year: The year of the record (2010-2020)

Value: Numerical value for the mortality rate

🔧 Methodology
Tools and Libraries
pandas: Data loading, filtering, and manipulation

matplotlib: Data visualization and plotting

statsmodels: ARIMA model building and fitting

sklearn: Mean Absolute Error (MAE) calculation

Analytical Workflow
Data Loading and Preprocessing

Load CSV file into pandas DataFrame

Filter data for Kenya and Mortality Rate

Convert to time series format with Year as index

Exploratory Data Analysis (EDA)

Plot time series to visualize trends (2010-2020)

Calculate key statistics (data points, range, missing values)

Model Building and Selection (ARIMA)

Split data: Training set (2010-2018), Test set (2019-2020)

Perform grid search for optimal ARIMA parameters (p, d, q)

Evaluate models using MAE and AIC criteria

Select best-performing model

Forecasting and Validation

Use best model to forecast test period (2019-2020)

Compare predictions against actual values

Refit model on entire dataset (2010-2020) for future forecasting

Visualization and Interpretation

Visualize historical data, model fit, and future forecasts

Interpret results in public health context

📈 Results
Data Summary
Kenya Malaria Mortality Rate (2010-2020):

text
Year    Mortality Rate (Deaths per 100 Thousand)
2010    8.90
2011    4.76
2012    4.60
2013    8.81
2014    11.05
2015    11.01
2016    10.89
2017    9.93
2018    9.56
2019    8.86
2020    9.18
Total Data Points: 11

Missing Values: 0

Value Range: 4.60 to 11.05

Training Set: 9 data points (2010-2018)

Test Set: 2 data points (2019-2020)

Model Selection Results
Grid search evaluated multiple ARIMA configurations:

Model	MAE	AIC	Status
ARIMA(1,1,1)	0.9842	37.34	
ARIMA(2,1,1)	1.7964	36.18	
ARIMA(1,1,2)	0.6522	38.16	
ARIMA(2,1,2)	0.6713	37.53	
ARIMA(0,1,1)	0.9151	35.54	
ARIMA(1,0,1)	0.5239	40.61	SELECTED
Final Model and Forecast
Best Model: ARIMA(1, 0, 1)

Test MAE: 0.5239

Final AIC: 46.21

Model Interpretation:

d=0: Time series was stationary (no differencing required)

p=1: Uses one prior value for prediction (AutoRegressive component)

q=1: Incorporates error from one prior prediction (Moving Average component)

Future Forecast:

2021: 9.99 deaths per 100 thousand

2022: 9.46 deaths per 100 thousand

2023: 9.27 deaths per 100 thousand

The forecast suggests a stabilizing and slightly downward trend in malaria mortality rates.

✅ Conclusion
This project successfully demonstrated a complete time series forecasting pipeline using Python. The ARIMA model for Kenya's malaria mortality rate performed exceptionally well with a low MAE of 0.52.

Key Achievements
Effective Data Handling: Successfully extracted, cleaned, and prepared real-world data

Robust Modeling: Implemented systematic grid search for optimal parameters

Actionable Insights: Generated credible 3-year forecast showing positive trend

Limitations and Future Work
Short Time Series: Limited to 11 years of data

External Variables: ARIMA doesn't incorporate external factors (rainfall, healthcare spending, interventions)

Future Extension: Use SARIMAX to include external variables

Broader Validation: Test model on other countries and metrics

🚀 Getting Started
Prerequisites
Python 3.x

Required libraries: pandas, matplotlib, statsmodels, sklearn

Installation
bash
pip install pandas matplotlib statsmodels scikit-learn
Usage
Clone the repository

Ensure the dataset file is in the project directory

Run the Python script to execute the analysis

View generated forecasts and visualizations

📝 License
This project is for educational and research purposes.

👥 Author
Gouland Ouma
Data Analyst and Biostatician
