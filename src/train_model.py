from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import joblib

def train_model(df, model_type='random_forest'):
    X = df.drop(columns=['cnt'])
    y = df['cnt']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = LinearRegression()

    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, f'../models/{model_type}_model.pkl')

    return model, X_test, y_test
