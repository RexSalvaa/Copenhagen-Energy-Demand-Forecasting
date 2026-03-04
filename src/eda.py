import pandas as pd
import requests
import json


def fetch_energy_data(limit=1000):
    url = 'https://api.energidataservice.dk/dataset/ConsumptionConsumerCategoryHour'

    # Copenhagen Region
    filter_dict = {"RegionName": "Region Hovedstaden"}

    params = {
        'limit': limit,
        'filter': json.dumps(filter_dict)
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        df = pd.DataFrame(response.json().get('records', []))
        # Conversion immédiate en datetime pour pouvoir travailler
        df['TimeDK'] = pd.to_datetime(df['TimeDK'])
        return df
    return None


import matplotlib.pyplot as plt
import seaborn as sns


def plot_consumption(df):
    plt.figure(figsize=(12, 6))

    # We create a line plot: Time on X-axis, Consumption on Y-axis
    # We use 'hue' to draw a different colored line for each category
    sns.lineplot(data=df, x='TimeDK', y='ConsumptionkWh', hue='ConsumerCategory2')

    plt.title('Energy Consumption Heartbeat in Copenhagen (Region Hovedstaden)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_predictions(val_df, predictions):
    plt.figure(figsize=(12, 6))

    # On prend une seule catégorie pour que ce soit lisible (ex: Erhverv)
    mask = val_df['cat_Erhverv'] == 1

    plt.plot(val_df.loc[mask, 'TimeDK'], val_df.loc[mask, 'ConsumptionkWh'],
             label='Réel (Actual)', color='blue', linewidth=2)
    plt.plot(val_df.loc[mask, 'TimeDK'], predictions[mask],
             label='Prédiction (XGBoost)', color='red', linestyle='--', linewidth=2)

    plt.title('Performance du Modèle : Réel vs Prédiction (Secteur Entreprises)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# In your main block:
if __name__ == "__main__":
    df = fetch_energy_data(limit=2000)  # Get a bit more data for a better graph
    if df is not None:
        plot_consumption(df)