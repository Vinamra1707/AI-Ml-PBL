import pickle
import numpy as np
import pandas as pd

model_dict = {
    1: ("Linear Regression", "linear_regression_model.pkl"),
    2: ("Ridge Regression", "ridge_regression_model.pkl"),
    3: ("Decision Tree", "decision_tree_model.pkl"),
    4: ("Random Forest", "random_forest_model.pkl")
}


def load_pickle_file(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

#Load all the saved models
def load_model(model_choice):
    model_name, model_file = model_dict[model_choice]
    model = load_pickle_file(model_file)
    scaler = load_pickle_file("scaler.pkl")
    feature_names = load_pickle_file("feature_names.pkl")
    return model_name, model, scaler, feature_names


def load_results():
    return pd.read_csv("model_comparison_results.csv")

#Display the beat model of along with its evaluated metrics
def display_best_model(results_df):
    best_row = results_df.loc[results_df["R2"].idxmax()]

    best_model = best_row["Model"]
    mae = best_row["MAE"]
    rmse = best_row["RMSE"]
    mse = rmse ** 2
    r2 = best_row["R2"]

    print("\n===== BEST MODEL =====")
    print(f"Best Model : {best_model}")
    print(f"R2 Score   : {r2:.4f}")
    print(f"RMSE       : {rmse:.4f}")
    print(f"MSE        : {mse:.4f}")
    print(f"MAE        : {mae:.4f}")

#function to display the evaluated metrics of selected model
def display_selected_model_metrics(results_df, model_name):
    row = results_df[results_df["Model"] == model_name].iloc[0]

    mae = row["MAE"]
    rmse = row["RMSE"]
    mse = rmse ** 2
    r2 = row["R2"]

    print("\n===== SELECTED MODEL METRICS ON TEST DATA=====")
    print(f"Model      : {model_name}")
    print(f"R2 Score   : {r2:.4f}")
    print(f"RMSE       : {rmse:.4f}")
    print(f"MSE        : {mse:.4f}")
    print(f"MAE        : {mae:.4f}")

#Input Feature Values from the user
def get_feature_input(feature_names):
    values = []

    print("\nEnter feature values:")
    for feature in feature_names:
        while True:
            try:
                value = float(input(f"{feature}: "))
                values.append(value)
                break
            except ValueError:
                print("Invalid input. Please enter a numeric value.")

    return pd.DataFrame([values], columns=feature_names)

#Dispay the Evaluated Metrics of the best model
def predict_energy():
    results_df = load_results()

    display_best_model(results_df)

    print("\n===== SELECT MODEL =====")
    print("1. Linear Regression")
    print("2. Ridge Regression")
    print("3. Decision Tree")
    print("4. Random Forest")

    while True:
        try:
            choice = int(input("Select model: "))
            if choice in model_dict:
                break
            else:
                print("Please select a number from 1 to 4.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    model_name, model, scaler, feature_names = load_model(choice)

    input_df = get_feature_input(feature_names)
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]


#Predict the energy usage acoording to the input values
    print(f"\nPredicted Energy Usage (kWh): {prediction:.4f}")



#display the evaluated metrics of selected model
    display_selected_model_metrics(results_df, model_name)

    check_actual = input("\nDo you want to compare with actual value? (yes/no): ").strip().lower()

    if check_actual == "yes":
        try:
            actual = float(input("Enter actual Energy Usage (kWh): "))
            abs_error = abs(actual - prediction)

            print("\n===== SINGLE VALUE COMPARISON =====")
            print(f"Actual Value     : {actual:.4f}")
            print(f"Predicted Value  : {prediction:.4f}")
            print(f"Absolute Error   : {abs_error:.4f}")

            if abs_error <= 5:
                print("Result           : Prediction is close")
            else:
                print("Result           : Prediction is not close")
        except ValueError:
            print("Invalid actual value entered.")


predict_energy()