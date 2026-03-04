import pandas as pd


def engineer_features(df):
    df = df.copy()

    # Ensure chronological order per category
    df = df.sort_values(['ConsumerCategory2', 'TimeDK'])

    # 1. Temporal Features (Same as before)
    df['hour'] = df['TimeDK'].dt.hour
    df['day_of_week'] = df['TimeDK'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    # 2. STABLE LAGS (The "No-Cheating" Anchors)
    # What happened exactly 24 hours ago?
    df['lag_24h'] = df.groupby('ConsumerCategory2')['ConsumptionkWh'].shift(24)

    # What happened exactly 7 days ago? (Very powerful for weekends)
    df['lag_7d'] = df.groupby('ConsumerCategory2')['ConsumptionkWh'].shift(24 * 7)

    # 3. ROLLING STATS (Context)
    # Average consumption over the last 24h period ending yesterday
    df['rolling_mean_24h'] = df.groupby('ConsumerCategory2')['lag_24h'].transform(
        lambda x: x.rolling(window=24).mean()
    )

    # 4. Encodage & Cleaning
    df = pd.get_dummies(df, columns=['ConsumerCategory2'], prefix='cat')

    # We drop the rows that don't have enough history for the 7-day lag
    df = df.dropna()

    return df


def split_data(df, test_days=2):
    """
    Sépare les données en Training et Validation.
    On prend les 'test_days' derniers jours pour la validation.
    """
    # On s'assure que les données sont triées par temps
    df = df.sort_values('TimeDK')

    # Calcul de la date de coupure
    last_date = df['TimeDK'].max()
    cutoff_date = last_date - pd.Timedelta(days=test_days)

    train_df = df[df['TimeDK'] < cutoff_date]
    test_df = df[df['TimeDK'] >= cutoff_date]

    return train_df, test_df