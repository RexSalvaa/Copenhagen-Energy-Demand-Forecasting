import pandas as pd


def engineer_features(df):
    df = df.copy()

    df = df.sort_values(['ConsumerCategory2', 'TimeDK'])

    df['hour'] = df['TimeDK'].dt.hour
    df['day_of_week'] = df['TimeDK'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    df['lag_24h'] = df.groupby('ConsumerCategory2')['ConsumptionkWh'].shift(24)

    df['lag_7d'] = df.groupby('ConsumerCategory2')['ConsumptionkWh'].shift(24 * 7)

    df['rolling_mean_24h'] = df.groupby('ConsumerCategory2')['lag_24h'].transform(
        lambda x: x.rolling(window=24).mean()
    )

    df = pd.get_dummies(df, columns=['ConsumerCategory2'], prefix='cat')

    df = df.dropna()

    return df


def split_data(df, test_days=2):

    df = df.sort_values('TimeDK')
    last_date = df['TimeDK'].max()
    cutoff_date = last_date - pd.Timedelta(days=test_days)

    train_df = df[df['TimeDK'] < cutoff_date]
    test_df = df[df['TimeDK'] >= cutoff_date]

    return train_df, test_df