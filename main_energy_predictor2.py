import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# --------- MODEL OPTIONS ---------
model_dict = {
    1: ("Linear Regression", "linear_regression_model.pkl"),
    2: ("Ridge Regression", "ridge_regression_model.pkl"),
    3: ("Decision Tree", "decision_tree_model.pkl"),
    4: ("Random Forest", "random_forest_model.pkl")
}

# --------- LOAD SAVED FILES ---------
def load_model(model_choice):
    model_name, model_file = model_dict[model_choice]
    model = pickle.load(open(model_file, "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    feature_names = pickle.load(open("feature_names.pkl", "rb"))
    return model_name, model, scaler, feature_names

# --------- INPUT HANDLING ---------
def get_feature_input(feature_names):
    values = []
    for f in feature_names:
        try:
            val = float(input(f"Enter {f}: "))
            values.append(val)
        except:
            print("Invalid input! Try again.")
            return None
    return pd.DataFrame([values], columns=feature_names)

# --------- SHOW METRICS ---------
def show_metrics(model_name, model, scaler):
    df = pd.read_csv("cleaned_dataset.csv")
    X = df.drop(columns=["Timestamp", "Date", "Energy_Usage_kWh", "Month"])
    y = df["Energy_Usage_kWh"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n===== SELECTED MODEL EVALUATION METRICS =====")
    print(f"Model : {model_name}")
    print(f"MAE   : {mae:.6f}")
    print(f"RMSE  : {rmse:.6f}")
    print(f"R2    : {r2:.6f}")

# --------- PREDICTION ---------
def predict_energy(model_choice):
    model_name, model, scaler, feature_names = load_model(model_choice)

    show_metrics(model_name, model, scaler)
    
    

    print(f"\nSelected Model: {model_name}")
    print("\nEnter values for prediction:")

    data = get_feature_input(feature_names)
    if data is None:
        return

    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)[0]

    print(f"\nPredicted Energy Usage (kWh): {prediction:.4f}")

    choice = input("\nDo you want to compare with actual value? (yes/no): ").lower()
    if choice == "yes":
        actual = float(input("Enter actual Energy Usage (kWh): "))
        error = abs(actual - prediction)

        print(f"Actual Value      : {actual:.4f}")
        print(f"Predicted Value   : {prediction:.4f}")
        print(f"Absolute Error    : {error:.4f}")
        

        if error <= 5:
            print("Result            : Prediction is close")
        else:
            print("Result            : Prediction is not close")

# --------- MAIN MENU ---------
def menu():
    while True:
        print("\n===============================")
        print("ENERGY USAGE PREDICTION SYSTEM")
        print("===============================")
        print("1. Use Linear Regression")
        print("2. Use Ridge Regression")
        print("3. Use Decision Tree")
        print("4. Use Random Forest")
        print("5. Exit")

        try:
            choice = int(input("Select model: "))
        except:
            print("Invalid input!")
            continue

        if choice == 5:
            print("Exiting...")
            break

        if choice not in model_dict:
            print("Invalid choice!")
            continue

        predict_energy(choice)

# --------- RUN ---------
menu()
