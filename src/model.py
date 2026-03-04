import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score


def train_model(train_df):
    """
    Entraîne un modèle XGBoost pour capturer les relations non-linéaires.
    """
    # On garde uniquement les nombres et on drop la cible
    X_train = train_df.select_dtypes(include=['number']).drop(columns=['ConsumptionkWh'])
    y_train = train_df['ConsumptionkWh']

    # Initialisation du modèle XGBoost
    # n_estimators=100 : nombre d'arbres de décision à construire
    # learning_rate=0.1 : vitesse d'apprentissage
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, val_df):
    # Même logique pour la validation
    X_val = val_df.select_dtypes(include=['number']).drop(columns=['ConsumptionkWh'])
    y_val = val_df['ConsumptionkWh']

    predictions = model.predict(X_val)

    mae = mean_absolute_error(y_val, predictions)
    r2 = r2_score(y_val, predictions)

    return mae, r2, predictions