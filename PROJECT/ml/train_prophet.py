import pandas as pd
from prophet import Prophet
from prophet.serialize import model_to_json
import matplotlib.pyplot as plt

def train_prophet(input_file, model_file, variable):
    """
    This function loads the preprocessed data, trains a Prophet model, and saves it to a file.
    """
    # Load the preprocessed data
    df = pd.read_csv(input_file, index_col='date', parse_dates=True)

    # Prepare the data for Prophet
    df_prophet = df.reset_index()[['date', variable]].rename(columns={'date': 'ds', variable: 'y'})

    # Create and train the model
    model = Prophet()
    model.fit(df_prophet)

    # Save the model
    with open(model_file, 'w') as f:
        f.write(model_to_json(model))

    return model

def make_future_forecast(model, periods, freq='D'):
    """
    This function makes a future forecast using the trained model.
    """
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast

if __name__ == '__main__':
    input_file = 'c:\\HACKATHON\\PROJECT\\data\\preprocessed_nyc_weather_data.csv'
    model_file = 'c:\\HACKATHON\\PROJECT\\src\\ml\\prophet_model.json'
    variable = 'T2M'

    # Train the model
    model = train_prophet(input_file, model_file, variable)

    # Make a future forecast
    forecast = make_future_forecast(model, periods=365 * 5) # 5 years

    # Plot the forecast
    fig = model.plot(forecast)
    plt.title(f'Forecast for {variable}')
    plt.xlabel('Date')
    plt.ylabel(variable)
    plt.show()