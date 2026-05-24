import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#Loading Cleaned Dataset
df = pd.read_csv("cleaned_dataset.csv")
X = df.drop(columns=["Timestamp", "Date", "Energy_Usage_kWh", "Month"])
y = df["Energy_Usage_kWh"]

# --------- SPLIT + SCALE ---------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


#performing scaling on data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



#Model Trainnig
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)

ridge_grid = GridSearchCV(
    Ridge(),
    {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
    cv=5,
    scoring="neg_mean_absolute_error"
)
ridge_grid.fit(X_train_scaled, y_train)
ridge_model = ridge_grid.best_estimator_

decision_tree_model = DecisionTreeRegressor(random_state=42)
decision_tree_model.fit(X_train_scaled, y_train)

random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest_model.fit(X_train_scaled, y_train)

#Saving model locally using pickle module
pickle.dump(linear_model, open("linear_regression_model.pkl", "wb"))
pickle.dump(ridge_model, open("ridge_regression_model.pkl", "wb"))
pickle.dump(decision_tree_model, open("decision_tree_model.pkl", "wb"))
pickle.dump(random_forest_model, open("random_forest_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(list(X.columns), open("feature_names.pkl", "wb"))

#Evaluating model on test data
results = []
models = {
    "Linear Regression": linear_model,
    "Ridge Regression": ridge_model,
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model
}

for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    results.append([name, mae, rmse, r2])

results_df = pd.DataFrame(results, columns=["Model", "MAE", "RMSE", "R2"])
results_df.to_csv("model_comparison_results.csv", index=False)

print("===== MODEL COMPARISON RESULTS =====")
print(results_df.to_string(index=False))
