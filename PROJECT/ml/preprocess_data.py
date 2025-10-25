
import pandas as pd

def preprocess_data(input_file, output_file):
    """
    This function loads the raw weather data, preprocesses it, and saves it to a new CSV file.
    """
    # Load the data, skipping the header rows
    df = pd.read_csv(input_file, skiprows=13, names=['YEAR', 'MO', 'DY', 'T2M', 'PRECTOTCORR', 'RH2M'])

    # Replace -999 with NaN
    df.replace(-999, pd.NA, inplace=True)

    # Create a datetime column
    df['date'] = pd.to_datetime(df[['YEAR', 'MO', 'DY']].astype(str).agg('-'.join, axis=1))

    # Set the datetime column as the index
    df.set_index('date', inplace=True)

    # Drop the YEAR, MO, and DY columns
    df.drop(['YEAR', 'MO', 'DY'], axis=1, inplace=True)

    # Save the preprocessed data
    df.to_csv(output_file)

if __name__ == '__main__':
    input_file = 'c:\\HACKATHON\\PROJECT\\data\\nyc_weather_data_2020_2024.csv'
    output_file = 'c:\\HACKATHON\\PROJECT\\data\\preprocessed_nyc_weather_data.csv'
    preprocess_data(input_file, output_file)

