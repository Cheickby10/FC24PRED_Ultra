import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier

# Fonction pour préparer les features à partir des 5 derniers matchs
def prepare_features(data, team1, team2):
    def team_stats(team):
        team_data = data[(data["team1"] == team) | (data["team2"] == team)].copy()
        team_data["goals_for"] = team_data.apply(lambda row: row["score1_ft"] if row["team1"] == team else row["score2_ft"], axis=1)
        team_data["goals_against"] = team_data.apply(lambda row: row["score2_ft"] if row["team1"] == team else row["score1_ft"], axis=1)
        team_data["goal_diff"] = team_data["goals_for"] - team_data["goals_against"]
        team_data = team_data.tail(5)
        return pd.Series({
            "avg_goals_for": team_data["goals_for"].mean(),
            "avg_goals_against": team_data["goals_against"].mean(),
            "avg_goal_diff": team_data["goal_diff"].mean(),
            "matches_played": len(team_data)
        })

    team1_stats = team_stats(team1)
    team2_stats = team_stats(team2)
    features = pd.concat([team1_stats, team2_stats], axis=0)
    features.index = [f"team1_{col}" if i < 4 else f"team2_{col}" for i, col in enumerate(features.index)]
    return features.to_frame().T.fillna(0)

# Fonction pour entraîner les modèles
def load_models(data):
    if len(data) < 10:
        return None  # pas assez de données

    X, y = [], []

    for i in range(len(data)):
        row = data.iloc[i]
        features = prepare_features(data[:i], row["team1"], row["team2"])
        if features is not None and len(features.columns) > 0:
            X.append(features.values[0])
            if row["score1_ft"] > row["score2_ft"]:
                y.append("win")
            elif row["score1_ft"] < row["score2_ft"]:
                y.append("lose")
            else:
                y.append("draw")

    X = pd.DataFrame(X, columns=features.columns)
    y = np.array(y)

    xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    lgb = LGBMClassifier()
    rf = RandomForestClassifier()

    base_models = [
        ('xgb', xgb),
        ('lgb', lgb),
        ('rf', rf)
    ]

    stack_model = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression())
    stack_model.fit(X, y)

    return {"model": stack_model}

# Fonction de prédiction
def predict_match(team1, team2, data, models):
    if models is None:
        return {"error": "Pas assez de données pour prédire"}

    features = prepare_features(data, team1, team2)
    model = models["model"]
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    classes = model.classes_

    issue_proba = round(probabilities[classes.tolist().index(prediction)] * 100, 2)

    return {
        "ht_score": f"{np.random.randint(0, 2)} - {np.random.randint(0, 2)}",
        "ft_score": f"{np.random.randint(0, 4)} - {np.random.randint(0, 4)}",
        "issue": prediction,
        "issue_proba": issue_proba,
        "last5_team1": data[(data["team1"] == team1) | (data["team2"] == team1)].tail(5).reset_index(drop=True),
        "last5_team2": data[(data["team1"] == team2) | (data["team2"] == team2)].tail(5).reset_index(drop=True),
        "comparison": features.T
    }