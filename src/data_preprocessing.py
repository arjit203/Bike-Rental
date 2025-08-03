import pandas as pd

def preprocess_data(filepath):
    df = pd.read_csv(filepath)

    df['dteday'] = pd.to_datetime(df['dteday'])

    df['day'] = df['dteday'].dt.day
    df['month'] = df['dteday'].dt.month
    df['weekday'] = df['dteday'].dt.weekday


    # Drop unwanted columns
    df = df.drop(columns=['instant', 'dteday', 'casual', 'registered'])

    return df
