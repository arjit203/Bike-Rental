from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("Mean Absolute Error:", mae)
    print("Root Mean Squared Error:", rmse)
    print("R² Score:", r2)

    # Plot actual vs predicted
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.legend()
    plt.title("Actual vs Predicted Rentals")
    plt.savefig('../visuals/actual_vs_predicted.png')
    plt.show()
