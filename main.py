from src.eda import fetch_energy_data, plot_predictions
from src.processing import engineer_features, split_data
from src.model import train_model, evaluate_model
import pandas as pd

# In your main.py, you can now do this:
if __name__ == "__main__":
    # 1. Fetch enough data (7 days lag requires at least ~200+ hours of history)
    raw = fetch_energy_data(limit=18000)
    processed = engineer_features(raw)

    # Split the last day for testing
    train_set, val_set = split_data(processed, test_days=3)

    # 2. Train model
    model = train_model(train_set)

    # 3. DIRECT PREDICTION (Honest & Stable)
    # The model now uses yesterday's data (lag_24h) to predict today.
    # This is exactly how real utilities plan the grid.
    X_val = val_set.select_dtypes(include=['number']).drop(columns=['ConsumptionkWh'])
    predictions = model.predict(X_val)

    # 4. Results
    from sklearn.metrics import r2_score, mean_absolute_error

    r2 = r2_score(val_set['ConsumptionkWh'], predictions)
    mae = mean_absolute_error(val_set['ConsumptionkWh'], predictions)

    print(f"--- STABLE DAILY FORECAST ---")
    print(f"Honest R2 Score: {r2:.2f}")
    print(f"MAE: {mae:.2f} kWh")

    # 5. Visualisation
    plot_predictions(val_set, predictions)