import pandas as pd
import requests
import json


def fetch_energy_data(limit=1000):
    url = 'https://api.energidataservice.dk/dataset/ConsumptionConsumerCategoryHour'

    filter_dict = {"RegionName": "Region Hovedstaden"}

    params = {
        'limit': limit,
        'filter': json.dumps(filter_dict)
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        df = pd.DataFrame(response.json().get('records', []))
        df['TimeDK'] = pd.to_datetime(df['TimeDK'])
        return df
    return None


import matplotlib.pyplot as plt
import seaborn as sns


def plot_consumption(df):
    plt.figure(figsize=(12, 6))

    sns.lineplot(data=df, x='TimeDK', y='ConsumptionkWh', hue='ConsumerCategory2')

    plt.title('Energy Consumption Heartbeat in Copenhagen (Region Hovedstaden)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_predictions(val_df, predictions):
    plt.figure(figsize=(12, 6))

    mask = val_df['cat_Erhverv'] == 1

    plt.plot(val_df.loc[mask, 'TimeDK'], val_df.loc[mask, 'ConsumptionkWh'],
             label='Reall (Actual)', color='blue', linewidth=2)
    plt.plot(val_df.loc[mask, 'TimeDK'], predictions[mask],
             label='Prediction (XGBoost)', color='red', linestyle='--', linewidth=2)

    plt.title('Model Performance: Real vs Prediction (Erhverv Sector)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    df = fetch_energy_data(limit=2000)
    if df is not None:
        plot_consumption(df)