import pandas as pd

def preprocess_data(filepath):
    df = pd.read_csv(filepath)

    # Convert date to datetime
    df['dteday'] = pd.to_datetime(df['dteday'])

    # Extract day, month, weekday
    df['day'] = df['dteday'].dt.day
    df['month'] = df['dteday'].dt.month
    df['weekday'] = df['dteday'].dt.weekday

    # Drop unwanted columns
    df = df.drop(columns=['instant', 'dteday', 'casual', 'registered'])

    # No missing values to handle

    return df
