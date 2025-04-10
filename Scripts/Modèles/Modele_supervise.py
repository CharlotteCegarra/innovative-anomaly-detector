# -----------------------------------------------------------------------------
# SCRIPT DE CLASSIFICATION SUPERVISÉE – PREDICTION D’ANOMALIE
# -----------------------------------------------------------------------------
# * Objectif :
#     Identifier automatiquement les nœuds instables à partir des caractéristiques agrégées
#     à l’échelle `olt_peag` en entraînant un modèle supervisé.

# * Deux variantes du pipeline sont présentées :

# * Variante 1 – Définition de la cible par seuil robuste (MAD) :
#     - La cible binaire (`target`) est définie en comparant `weighted_avg_dns_time`
#       à un seuil dynamique : médiane + 1.5 * MAD
#     - Un nœud est considéré comme instable si sa latence dépasse ce seuil

# * Variante 2 – Modélisation supervisée :
#     - Les features utilisées sont :
#         * Variabilité DNS (std)
#         * Nombre de clients
#         * Moyennes et variabilités de latence scoring et score scoring
#     - Séparation train/test
#     - Modèle RandomForest avec équilibrage des classes (`class_weight='balanced'`)
#     - Évaluation du modèle : classification report + matrice de confusion
#     - Interprétation du modèle avec SHAP :
#         * Identification des variables les plus influentes via un bar plot
#         * Visualisation des contributions pour un échantillon donné

# * Sorties :
#     - Rapport d’évaluation du modèle
#     - Visualisation SHAP des top variables contributives
# -----------------------------------------------------------------------------


import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm




logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if pd.api.types.is_numeric_dtype(grouped['weighted_avg_dns_time']):
    grouped = grouped.dropna(subset=['weighted_avg_dns_time'])
    median_value = grouped['weighted_avg_dns_time'].median()
    mad_value = np.median(np.abs(grouped['weighted_avg_dns_time'] - median_value))
    threshold = median_value + 1.5 * mad_value
    logger.info(f"Seuil calculé pour weighted_avg_dns_time: {threshold:.3f}")
else:
    logger.error("La colonne 'weighted_avg_dns_time' n'est pas numérique.")
    threshold = None

if threshold is not None:
    target = (grouped['weighted_avg_dns_time'] > threshold).astype(int)
    logger.info(f"Répartition de la cible :\n{target.value_counts()}")

    features = grouped[['weighted_std_dns_time', 'nb_client_total', 'weighted_avg_latence_scoring',
                    'weighted_std_latence_scoring', 'weighted_avg_score_scoring',
                    'weighted_std_score_scoring']].copy()
    features = features.fillna(features.median())

    logger.info("Séparation train/test en cours...")
    for _ in tqdm(range(1), desc="Séparation train/test"):
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
    logger.info(f"Train set : {X_train.shape[0]} échantillons, Test set : {X_test.shape[0]} échantillons.")

    logger.info("Entraînement du modèle RandomForest avec class_weight='balanced'...")
    rf_model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    rf_model.fit(X_train, y_train)
    logger.info("Modèle RandomForest entraîné.")

    y_pred = rf_model.predict(X_test)
    logger.info("Prédictions effectuées sur le set de test.")

    logger.info("\nRapport de Classification :")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Prédiction')
    plt.ylabel('Réel')
    plt.title('Matrice de Confusion')
    plt.tight_layout()
    plt.show()

    logger.info("Calcul des valeurs SHAP...")
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test)

    shap.initjs()
    sample_index = 0
    shap.bar_plot(shap_values[1][sample_index], feature_names=X_test.columns, max_display=5)
    plt.title("Top 5 variables influentes - SHAP bar plot")
    plt.show()
