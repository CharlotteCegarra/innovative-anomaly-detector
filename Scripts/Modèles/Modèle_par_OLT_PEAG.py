# -----------------------------------------------------------------------------
# SCRIPT DE DÉTECTION D’ANOMALIES SUR LE CROISEMENT OLT-PEAG
# -----------------------------------------------------------------------------
# * Objectif :
#     Appliquer des modèles d’Isolation Forest sur les données agrégées par couple `olt_peag`
#     afin de détecter des comportements anormaux dans les métriques réseau (DNS, latence, scoring).

# * Deux variantes du pipeline sont présentées :

# * Variante 1 – Détection d’anomalies par test individuel :
#     - Trois variables de test sont analysées séparément :
#         * weighted_avg_dns_time
#         * weighted_avg_latence_scoring
#         * weighted_avg_score_scoring
#     - Pour chaque variable :
#         - Imputation des valeurs manquantes (médiane)
#         - Modèle IsolationForest entraîné
#         - Prédiction des anomalies par batch
#         - Colonne d'anomalie ajoutée au DataFrame (`anomaly_<test>`)

# * Variante 2 – Détection d’anomalies multivariée globale :
#     - Utilise toutes les métriques pondérées et leurs écarts-types :
#         * Temps DNS, latence scoring, score scoring + leurs std
#         * Nombre de tests et nombre de clients
#     - Vérifie la présence des colonnes nécessaires
#     - Applique un modèle IsolationForest multivarié
#     - Prédiction des anomalies en batch
#     - Colonne `anomaly` ajoutée au DataFrame global

# * Sorties :
#     - DataFrame enrichi de colonnes d’anomalie par variable (variante 1) ou globale (variante 2)
#     - Affichage du nombre total d’anomalies détectées par modèle
# -----------------------------------------------------------------------------


import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from tqdm import tqdm




# Modèle par croisement olt et peag pour chaque test

test_variables = [
    "weighted_avg_dns_time", "weighted_avg_latence_scoring", "weighted_avg_score_scoring"]

def detect_anomalies_per_test(df, test_variable, batch_size=1000):
    if df.empty:
        print("Le DataFrame est vide. Aucun modèle ne sera entraîné.")
        return None, df

    if test_variable not in df.columns:
        print(f"La colonne '{test_variable}' est manquante dans le DataFrame.")
        return None, df

    df[test_variable] = df[test_variable].fillna(df[test_variable].median())
    features = df[[test_variable]]

    if features.isnull().any().any():
        print(f"La colonne '{test_variable}' contient encore des NaN après le remplissage.")
        return None, df

    if features.shape[0] < 1:
        print(f"Pas assez de données pour entraîner le modèle pour le test '{test_variable}'.")
        return None, df

    try:
        model = IsolationForest(contamination=0.05, random_state=42)
        model.fit(features)

        predictions = []
        indices = np.arange(features.shape[0])
        batch_indices = np.array_split(indices, np.ceil(features.shape[0] / batch_size))
        for batch in tqdm(batch_indices, desc=f"Prédiction des anomalies pour '{test_variable}'"):
            batch_features = features.iloc[batch]
            batch_preds = model.predict(batch_features)
            predictions.extend(batch_preds)

        df[f'anomaly_{test_variable}'] = predictions
        print(f"Nombre d'anomalies détectées pour '{test_variable}' : {sum(np.array(predictions) == -1)}")

        return model, df

    except ValueError as e:
        print(f"Erreur lors de l'entraînement du modèle pour '{test_variable}' : {e}")
        return None, df


for test_variable in test_variables:
    print(f"\nDétection des anomalies pour le test : '{test_variable}'")
    model, df_anomalies = detect_anomalies_per_test(grouped, test_variable, batch_size=100)


# Modèle par croisement olt et peag pour les trois tests 

numerical_variables = [
    "nb_test_dns", "nb_test_scoring", "weighted_avg_dns_time", "weighted_std_dns_time",
    "weighted_avg_latence_scoring", "weighted_std_latence_scoring",
    "weighted_avg_score_scoring", "weighted_std_score_scoring"
]

def detect_anomalies_olt_peag_batch(df, batch_size=1000):
    if df.empty:
        print("Le DataFrame est vide. Aucun modèle ne sera entraîné.")
        return None, df

    missing_columns = [col for col in numerical_variables if col not in df.columns]
    if missing_columns:
        print(f"Colonnes manquantes dans le DataFrame : {missing_columns}")
        return None, df

    df[numerical_variables] = df[numerical_variables].fillna(df[numerical_variables].median())
    features = df[numerical_variables]

    if features.isnull().any().any():
        print("Certaines colonnes contiennent encore des NaN après le remplissage.")
        return None, df

    if features.shape[0] < 1:
        print("Pas assez de données pour entraîner le modèle.")
        return None, df

    try:
        model = IsolationForest(contamination=0.05, random_state=42)
        model.fit(features)

        predictions = []
        indices = np.arange(features.shape[0])
        batch_indices = np.array_split(indices, np.ceil(features.shape[0] / batch_size))
        for batch in tqdm(batch_indices, desc="Prédiction des anomalies par batch"):
            batch_features = features.iloc[batch]
            batch_preds = model.predict(batch_features)
            predictions.extend(batch_preds)

        df['anomaly'] = predictions
        print(f"Nombre d'anomalies détectées dans le modèle `olt_peag` : {sum(np.array(predictions) == -1)}")

        return model, df

    except ValueError as e:
        print(f"Erreur lors de l'entraînement du modèle : {e}")
        return None, df

model_olt_peag, df_anomalies = detect_anomalies_olt_peag_batch(grouped, batch_size=100)
