from src.eda import fetch_energy_data, plot_predictions
from src.processing import engineer_features, split_data
from src.model import train_model, evaluate_model
import pandas as pd

if __name__ == "__main__":
    raw = fetch_energy_data(limit=5000)
    processed = engineer_features(raw)

    train_set, val_set = split_data(processed, test_days=3)

    model = train_model(train_set)

    X_val = val_set.select_dtypes(include=['number']).drop(columns=['ConsumptionkWh'])
    predictions = model.predict(X_val)

    from sklearn.metrics import r2_score, mean_absolute_error

    r2 = r2_score(val_set['ConsumptionkWh'], predictions)
    mae = mean_absolute_error(val_set['ConsumptionkWh'], predictions)

    print(f"--- STABLE DAILY FORECAST ---")
    print(f"Honest R2 Score: {r2:.2f}")
    print(f"MAE: {mae:.2f} kWh")

    plot_predictions(val_set, predictions)