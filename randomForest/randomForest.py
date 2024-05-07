import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv('sensor_raw2.csv')
window_size = 14

def extract_features(data):
    features = []
    labels = []
    driver_ratings = {}
    total_y_test = []  # To store y_test for all drivers
    total_y_pred = []  # To store y_pred for all drivers

    for driver_id, driver_data in data.groupby('DriverID'):
        X = []
        y = []
        for _, task_data in driver_data.groupby('Class'):
            for i in range(0, len(task_data) - window_size):
                window = task_data.iloc[i:i + window_size]
                feature_vector = [window['AccX'].mean(),
                                  window['AccY'].mean(),
                                  window['AccZ'].mean(),
                                  window['GyroX'].mean(),
                                  window['GyroY'].mean(),
                                  window['GyroZ'].mean()]
                X.append(feature_vector)
                y.append(window['Class'].values[0])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Append y_test and y_pred for the current driver to the total lists
        total_y_test.extend(y_test)
        total_y_pred.extend(y_pred)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy for {driver_id}: {accuracy}")
        driver_ratings[driver_id] = accuracy

    # Calculate and print overall accuracy
    overall_accuracy = accuracy_score(total_y_test, total_y_pred)
    print(f"\nOverall Model Accuracy: {overall_accuracy}")

    return driver_ratings

driver_ratings = extract_features(data)

for driver_id, accuracy in driver_ratings.items():
    if accuracy > 0.9:
        driver_rating = "Excellent"
    elif accuracy > 0.8:
        driver_rating = "Good"
    elif accuracy > 0.7:
        driver_rating = "Average"
    else:
        driver_rating = "Poor"

    print(f"{driver_id} Rating: {driver_rating}")