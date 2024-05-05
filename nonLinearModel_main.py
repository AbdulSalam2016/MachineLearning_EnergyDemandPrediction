
import seaborn as sns
import scipy.stats as stats
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import SVR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.close('all')  # Closes all figures
import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
pd.set_option('display.max_columns', None)

def load_and_prepare_data():
    data_path = 'Complete_Electricity_Demand_Supply_Simulated_Data.csv'
    data = pd.read_csv(data_path)
    data = data.drop('DateTime', axis=1)

    categorical_vars = ['Month', 'Day', 'Hour', 'Season', 'Holiday']
    data = pd.get_dummies(data, columns=categorical_vars, drop_first=True)
    data['net_supply'] = data['Solar_Supply_MW'] + data['Wind_Supply_MW'] - data['Demand_MW']
    data['Temperature_sqrd'] = data['Temperature']**2

    features = ['Temperature', 'Temperature_sqrd', 'Sunlight_Hours', 'Economic_Indicator'] + \
               [col for col in data.columns if col.startswith('Month_') or col.startswith('Day_') or col.startswith('Hour_') or col.startswith('Season_') or col.startswith('Holiday_')]
    X = data[features]
    y = data['net_supply']
    # print(data[features].columns)
    return X, y

def evaluate_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Decision Tree Regressor': DecisionTreeRegressor(),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=100),
        'Gradient Boosting Regressor': GradientBoostingRegressor(n_estimators=100),
        'Gaussian Process Regressor': GaussianProcessRegressor(),
        'MLP Neural Network': MLPRegressor(random_state=1, max_iter=500),
        'SVR': SVR()
    }
    best_r2 = float('-inf')
    best_model = None
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = (mse, r2)
        if r2 > best_r2:
            best_r2 = r2
            best_model = name
    print("Model performance:")
    for name, scores in results.items():
        print(f"{name} - MSE: {scores[0]:.2f}, R-squared: {scores[1]:.2f}")
    print(f"Best model: {best_model} with the highest R-squared of {best_r2:.2f}")
    pass

def perform_hyperparameter_tuning(X, y):
    mlp = MLPRegressor(max_iter=1000, random_state=42)
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (150,), (100, 50)],
        'activation': ['relu', 'tanh'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive'],
    }
    grid = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
    grid.fit(X, y)
    print("Best parameters found: ", grid.best_params_)
    return grid.best_estimator_

def save_model(model, filename='best_mlp_model.joblib'):
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def plot_results(model, X_test, y_test, y_pred):
    # Plotting Actual vs. Predicted values
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred)  # Corrected to use keyword arguments for x and y
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs. Predicted')
    plt.show()

    # Plotting the distribution of residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.title('Residuals Distribution')
    plt.show()

    # Q-Q plot for residuals
    plt.figure(figsize=(6, 6))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot of Residuals')
    plt.show()

    # Feature importance plot for models that support it
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(10, 6))
        plt.title('Feature Importances')
        sns.barplot(x=importances[indices], y=X_test.columns[indices], orient='h')
        plt.show()

def main():
    # Load and prepare data
    X, y = load_and_prepare_data()

    # Splitting the dataset for demo purposes
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Assume you might call evaluate_models here if needed
    evaluate_models(X, y)

    # Hyperparameter tuning specifically for MLP Neural Network
    nonlinear_best_mlp_model = perform_hyperparameter_tuning(X_train, y_train)
    save_model(nonlinear_best_mlp_model, 'nonlinear_best_mlp_model.joblib')  # Save the best MLP model

    # Use the best model to predict the test set
    y_pred = nonlinear_best_mlp_model.predict(X_test)
    plot_results(nonlinear_best_mlp_model, X_test, y_test, y_pred)  # Use tuned model to generate plots

if __name__ == "__main__":
    main()
