# --- BLOCK 1: Import Libraries ---
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# --- BLOCK 2: Load and Explore Data ---
file_path = "C:\\Users\\Admin\\Downloads\\openafrica-_-malaria-_-national_unit-data-infection-prevalence.csv"
df = pd.read_csv(file_path)
print("Dataset loaded successfully!")
print(df.info())
print("\nUnique Countries:", df['Name'].unique())
print("\nUnique Metrics:", df['Metric'].unique())

# --- BLOCK 3: Preprocess Data for a Specific Country/Metric ---
target_country = "Kenya"
target_metric = "Mortality Rate"

# Filter the dataset
df_filtered = df[(df['Name'] == target_country) & (df['Metric'] == target_metric)]

if df_filtered.empty:
    print(f"No data found for {target_country} - {target_metric}")
    print("Available combinations:")
    combinations = df.groupby(['Name', 'Metric']).size().reset_index(name='count')
    print(combinations.head(10))
else:
    # CRITICAL FIX: Create proper datetime index
    df_filtered = df_filtered.copy()
    df_filtered['datetime'] = pd.to_datetime(df_filtered['Year'], format='%Y')
    ts_data = df_filtered.set_index('datetime')['Value'].sort_index()
    
    # Remove any NaN values
    ts_data = ts_data.dropna()
    
    print(f"\nTime Series Data for {target_metric} in {target_country}:")
    print(ts_data)
    
    # Check if we have enough data
    if len(ts_data) < 5:
        print("Not enough data points for modeling. Need at least 5 years of data.")
    else:
        # --- BLOCK 4: Visualize the Time Series ---
        plt.figure(figsize=(10, 5))
        plt.plot(ts_data.index.year, ts_data.values, marker='o', linestyle='-')
        plt.title(f'{target_metric} in {target_country}')
        plt.xlabel('Year')
        plt.ylabel(f'{target_metric} ({df_filtered["Units"].iloc[0]})')
        plt.grid(True)
        plt.show()
        
        # --- BLOCK 5: Split Data ---
        # Use datetime-based splitting
        split_date = pd.to_datetime('2018-12-31')
        train = ts_data[ts_data.index <= split_date]
        test = ts_data[ts_data.index > split_date]
        
        print(f"\nTrain data points: {len(train)}")
        print(f"Test data points: {len(test)}")
        
        if len(test) == 0:
            print("No test data available. Using all data for final model.")
            train = ts_data
            test = None
        
        # --- BLOCK 6: Model Selection and Fitting ---
        best_mae = float('inf')
        best_model = None
        best_forecast = None
        best_params = None
        
        # Test different ARIMA configurations
        arima_configs = [
            (1, 1, 1),
            (2, 1, 1),
            (1, 1, 2),
            (2, 1, 2),
            (0, 1, 1),
            (1, 0, 1)
        ]
        
        print("\nTesting different ARIMA configurations:")
        
        for p, d, q in arima_configs:
            try:
                # Fit ARIMA model
                model = ARIMA(train, order=(p, d, q))
                fitted_model = model.fit()
                
                if test is not None and len(test) > 0:
                    # Forecast for test period
                    forecast = fitted_model.forecast(steps=len(test))
                    
                    # Handle potential NaN values
                    if not any(pd.isna(forecast)):
                        mae = mean_absolute_error(test, forecast)
                        print(f"  ARIMA({p},{d},{q}) - MAE: {mae:.4f}, AIC: {fitted_model.aic:.2f}")
                        
                        if mae < best_mae:
                            best_mae = mae
                            best_model = fitted_model
                            best_forecast = pd.Series(forecast, index=test.index)
                            best_params = (p, d, q)
                    else:
                        print(f"  ARIMA({p},{d},{q}) - Forecast contains NaN values, skipping")
                else:
                    # No test data, just fit and evaluate on AIC
                    print(f"  ARIMA({p},{d},{q}) - AIC: {fitted_model.aic:.2f}")
                    if best_model is None or fitted_model.aic < best_model.aic:
                        best_model = fitted_model
                        best_params = (p, d, q)
                        
            except Exception as e:
                print(f"  ARIMA({p},{d},{q}) - Error: {str(e)[:50]}...")
                continue
        
        if best_model is None:
            print("No successful model found. Using simple ARIMA(1,1,1)...")
            model = ARIMA(train, order=(1, 1, 1))
            best_model = model.fit()
            best_params = (1, 1, 1)
        
        print(f"\nBest model: ARIMA{best_params}")
        
        # --- BLOCK 7: Evaluate on Test Data (if available) ---
        if test is not None and len(test) > 0 and best_forecast is not None:
            print(f"Test MAE: {best_mae:.4f}")
            
            # Plot evaluation results
            plt.figure(figsize=(12, 6))
            plt.plot(train.index.year, train.values, label='Training Data', color='blue', marker='o')
            plt.plot(test.index.year, test.values, label='Actual Test Data', color='green', marker='o')
            plt.plot(test.index.year, best_forecast.values, label=f'ARIMA{best_params} Forecast', color='red', linestyle='--', marker='s')
            plt.title(f'Model Evaluation: {target_metric} in {target_country}')
            plt.xlabel('Year')
            plt.ylabel(f'{target_metric} ({df_filtered["Units"].iloc[0]})')
            plt.legend()
            plt.grid(True)
            plt.show()
        
        # --- BLOCK 8: Final Model & Future Forecast ---
        # Fit on complete dataset
        final_model = ARIMA(ts_data, order=best_params)
        final_fitted = final_model.fit()
        
        # Forecast future years
        future_steps = 3  # Forecast 3 years ahead
        future_forecast = final_fitted.forecast(steps=future_steps)
        
        # Create proper future dates
        last_year = ts_data.index.max().year
        future_years = range(last_year + 1, last_year + future_steps + 1)
        
        # Handle potential NaN forecasts
        if not any(pd.isna(future_forecast)):
            print(f"\nFuture Forecast for {target_metric} in {target_country}:")
            for year, value in zip(future_years, future_forecast):
                print(f"  {year}: {value:.2f}")
            
            # --- BLOCK 9: Final Visualization ---
            plt.figure(figsize=(12, 6))
            plt.plot(ts_data.index.year, ts_data.values, label='Historical Data', color='blue', marker='o')
            plt.plot(future_years, future_forecast, label=f'Forecast ({last_year+1}-{last_year+future_steps})', 
                    color='red', linestyle='--', marker='s')
            plt.title(f'Future Forecast: {target_metric} in {target_country}')
            plt.xlabel('Year')
            plt.ylabel(f'{target_metric} ({df_filtered["Units"].iloc[0]})')
            plt.legend()
            plt.grid(True)
            plt.show()
            
            # Print model summary
            print(f"\nModel Summary:")
            print(f"Best ARIMA order: {best_params}")
            print(f"AIC: {final_fitted.aic:.2f}")
            
        else:
            print("Forecast contains NaN values. Model may not be suitable for this data.")
            print("Try different ARIMA parameters or check data quality.")

# --- ADDITIONAL: Data Quality Check ---
print(f"\nData Quality Check:")
print(f"Total data points: {len(ts_data)}")
print(f"Missing values: {ts_data.isna().sum()}")
print(f"Data range: {ts_data.index.min().year} to {ts_data.index.max().year}")
print(f"Value range: {ts_data.min():.2f} to {ts_data.max():.2f}")
