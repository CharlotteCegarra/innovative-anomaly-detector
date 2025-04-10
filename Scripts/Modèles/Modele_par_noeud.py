# -----------------------------------------------------------------------------
# DÉTECTION D’ANOMALIES PAR MODÈLE ISOLATION FOREST PAR NŒUD ET PAR TEST
# -----------------------------------------------------------------------------
# * Objectif :
#     - Appliquer un modèle d’anomalie Isolation Forest pour chaque nœud du réseau (peag_nro, olt_name, boucle)
#     - Détection sur plusieurs variables de mesure (latence DNS, scoring, etc.)
#
# * Fonctionnalités principales :
#     - Nettoyage et prétraitement des données (dates, valeurs manquantes)
#     - Agrégation temporelle par heure pour lisser les séries
#     - Traitement parallèle par batchs pour améliorer les performances
#     - Entraînement d’un modèle Isolation Forest par nœud, ajusté dynamiquement au volume de données
#     - Ajout des colonnes `anomaly` et `anomaly_score` aux résultats
#
# * Résultats :
#     - Fusion des résultats de détection d’anomalies pour chaque nœud
#     - Résumés statistiques :
#         - Nombre total d’anomalies
#         - Pourcentage global d’anomalies
#         - Pourcentage d’anomalies par nœud
#     - Impression des résultats détaillés et des top nœuds les plus instables
#
# * Deux variantes identiques du pipeline sont présentées :
#     - Une pour le traitement individuel de chaque type de regroupement (PEAG, OLT, boucle)
#     - Une seconde version répliquée avec la même logique
# -----------------------------------------------------------------------------


import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import logging



# Modèle par noeud et par test

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

numerical_variables = [
    'avg_dns_time', 'std_dns_time',
    'avg_latence_scoring', 'std_latence_scoring',
    'avg_score_scoring', 'std_score_scoring'
]

def preprocess_data(df, time_column):
    df_copy = df.copy()
    df_copy[time_column] = pd.to_datetime(df_copy[time_column], errors='coerce')
    df_copy = df_copy.sort_values(by=[time_column])
    df_copy = df_copy.dropna(subset=[time_column])
    for col in numerical_variables:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].fillna(df_copy[col].median())
    return df_copy

def process_node(node_tuple):
    node_name, node_data = node_tuple
    time_column = 'date_hour'
    try:
        if node_data.shape[0] < 2:
            logger.warning(f"Pas assez de données pour le nœud {node_name} ({node_data.shape[0]} observations), nœud ignoré.")
            return node_name, None

        node_data = node_data.set_index(time_column)
        agg_dict = {col: 'mean' for col in numerical_variables if col in node_data.columns}
        if not agg_dict:
            logger.warning(f"Aucune variable numérique disponible pour le nœud {node_name}, nœud ignoré.")
            return node_name, None

        resampled = node_data.resample('h').agg(agg_dict)

        if resampled.shape[0] < 2:
            logger.warning(f"Pas assez de données après rééchantillonnage pour {node_name} ({resampled.shape[0]} observations), nœud ignoré.")
            return node_name, None

        resampled = resampled.ffill().bfill()
        if resampled.isnull().sum().sum() > 0:
            for col in resampled.columns:
                if resampled[col].isnull().any():
                    resampled[col] = resampled[col].fillna(resampled[col].median())
        if resampled.isnull().sum().sum() > 0:
            logger.warning(f"Valeurs manquantes persistantes pour {node_name} après traitement, nœud ignoré.")
            return node_name, None

        contamination = max(0.01, min(0.05, 10 / resampled.shape[0]))
        model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
            max_samples='auto',
            n_jobs=-1
        )
        model.fit(resampled)
        anomaly_scores = model.decision_function(resampled)
        predictions = model.predict(resampled)

        resampled['anomaly'] = predictions
        resampled['anomaly_score'] = anomaly_scores

        num_anomalies = sum(predictions == -1)
        logger.info(f"Nœud {node_name}: {num_anomalies} anomalies détectées sur {resampled.shape[0]} observations ({(num_anomalies / resampled.shape[0]) * 100:.2f}%)")
        return node_name, resampled.reset_index()

    except Exception as e:
        logger.error(f"Erreur lors du traitement du nœud {node_name}: {str(e)}")
        return node_name, None

def detect_anomalies(df, node_column, time_column, use_parallel=True, max_workers=None, batch_size=100):
    models = {}
    anomalies_results = []

    df_processed = preprocess_data(df, time_column)
    grouped = df_processed.groupby(node_column)
    unique_nodes = list(grouped.groups.keys())
    logger.info(f"Traitement de {len(unique_nodes)} nœuds uniques.")

    if max_workers is None:
        max_workers = min(4, len(unique_nodes)) if len(unique_nodes) > 0 else 1

    node_tuples = [(node, grouped.get_group(node)) for node in unique_nodes]
    batches = [node_tuples[i:i+batch_size] for i in range(0, len(node_tuples), batch_size)]
    logger.info(f"Traitement des nœuds en {len(batches)} batchs de maximum {batch_size} nœuds chacun.")

    if use_parallel:
        for batch in tqdm(batches, desc="Traitement des batchs de nœuds"):
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(process_node, nt): nt[0] for nt in batch}
                for future in as_completed(futures):
                    node = futures[future]
                    try:
                        node_name, result = future.result()
                        if result is not None:
                            anomalies_results.append(result)
                        else:
                            logger.info(f"Nœud {node_name} ignoré durant le traitement.")
                    except Exception as e:
                        logger.error(f"Erreur lors du traitement parallèle du nœud {node}: {str(e)}")
    else:
        for batch in tqdm(batches, desc="Traitement des batchs de nœuds"):
            for nt in batch:
                node_name, result = process_node(nt)
                if result is not None:
                    anomalies_results.append(result)

    if anomalies_results:
        result_df = pd.concat(anomalies_results, ignore_index=True)
    else:
        result_df = pd.DataFrame()

    logger.info(f"Détection d'anomalies terminée. {len(result_df)} observations avec anomalies détectées.")
    return models, result_df

def analyze_anomalies(result_df, node_column=None):
    if result_df.empty:
        logger.warning("Aucune anomalie détectée ou résultats vides.")
        return None

    total_observations = len(result_df)
    total_anomalies = (result_df['anomaly'] == -1).sum()
    anomaly_percentage = (total_anomalies / total_observations) * 100

    summary = {
        'total_observations': total_observations,
        'total_anomalies': total_anomalies,
        'anomaly_percentage': anomaly_percentage
    }

    if node_column and node_column in result_df.columns:
        node_stats = result_df.groupby(node_column)['anomaly'].apply(
            lambda x: (x == -1).sum() / len(x) * 100
        ).sort_values(ascending=False)
        summary['node_anomaly_percentages'] = node_stats

    return summary

if __name__ == "__main__":
    try:
        models_peag, results_peag = detect_anomalies(df, 'peag_nro', 'date_hour', use_parallel=True, batch_size=100)
        models_olt, results_olt = detect_anomalies(df, 'olt_name', 'date_hour', use_parallel=True, batch_size=100)
        models_boucle, results_boucle = detect_anomalies(df, 'boucle', 'date_hour', use_parallel=True, batch_size=100)

        peag_analysis = analyze_anomalies(results_peag, 'peag_nro')
        if not results_peag.empty:
            print("\nAperçu des résultats PEAG:")
            print(results_peag.head())
        if peag_analysis:
            print("\nAnalyse des anomalies PEAG:")
            for key, value in peag_analysis.items():
                if key != 'node_anomaly_percentages':
                    print(f"{key}: {value}")
            if 'node_anomaly_percentages' in peag_analysis:
                print("\nPourcentage d'anomalies par nœud (top 5) PEAG:")
                print(peag_analysis['node_anomaly_percentages'].head(5))

        olt_analysis = analyze_anomalies(results_olt, 'olt_name')
        if not results_olt.empty:
            print("\nAperçu des résultats OLT:")
            print(results_olt.head())
        if olt_analysis:
            print("\nAnalyse des anomalies OLT:")
            for key, value in olt_analysis.items():
                if key != 'node_anomaly_percentages':
                    print(f"{key}: {value}")
            if 'node_anomaly_percentages' in olt_analysis:
                print("\nPourcentage d'anomalies par nœud (top 5) OLT:")
                print(olt_analysis['node_anomaly_percentages'].head(5))

        boucle_analysis = analyze_anomalies(results_boucle, 'boucle')
        if not results_boucle.empty:
            print("\nAperçu des résultats BOUCLE:")
            print(results_boucle.head())
        if boucle_analysis:
            print("\nAnalyse des anomalies BOUCLE:")
            for key, value in boucle_analysis.items():
                if key != 'node_anomaly_percentages':
                    print(f"{key}: {value}")
            if 'node_anomaly_percentages' in boucle_analysis:
                print("\nPourcentage d'anomalies par nœud (top 5) BOUCLE:")
                print(boucle_analysis['node_anomaly_percentages'].head(5))

    except Exception as e:
        logger.error(f"Erreur lors de l'exécution principale: {str(e)}")







# Modèle par noeud pour les trois ensemble de test 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

numerical_variables = [
    'avg_dns_time', 'std_dns_time',
    'avg_latence_scoring', 'std_latence_scoring',
    'avg_score_scoring', 'std_score_scoring'
]

def preprocess_data(df, time_column):
    df_copy = df.copy()
    df_copy[time_column] = pd.to_datetime(df_copy[time_column], errors='coerce')
    df_copy = df_copy.sort_values(by=[time_column])
    df_copy = df_copy.dropna(subset=[time_column])
    for col in numerical_variables:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].fillna(df_copy[col].median())
    return df_copy

def process_node(node_tuple):
    node_name, node_data = node_tuple
    time_column = 'date_hour'
    try:
        if node_data.shape[0] < 2:
            logger.warning(f"Pas assez de données pour le nœud {node_name} ({node_data.shape[0]} observations), nœud ignoré.")
            return node_name, None

        node_data = node_data.set_index(time_column)
        agg_dict = {col: 'mean' for col in numerical_variables if col in node_data.columns}
        if not agg_dict:
            logger.warning(f"Aucune variable numérique disponible pour le nœud {node_name}, nœud ignoré.")
            return node_name, None

        resampled = node_data.resample('h').agg(agg_dict)

        if resampled.shape[0] < 2:
            logger.warning(f"Pas assez de données après rééchantillonnage pour {node_name} ({resampled.shape[0]} observations), nœud ignoré.")
            return node_name, None

        resampled = resampled.ffill().bfill()
        if resampled.isnull().sum().sum() > 0:
            for col in resampled.columns:
                if resampled[col].isnull().any():
                    resampled[col] = resampled[col].fillna(resampled[col].median())
        if resampled.isnull().sum().sum() > 0:
            logger.warning(f"Valeurs manquantes persistantes pour {node_name} après traitement, nœud ignoré.")
            return node_name, None

        contamination = max(0.01, min(0.05, 10 / resampled.shape[0]))
        model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
            max_samples='auto',
            n_jobs=-1
        )
        model.fit(resampled)
        anomaly_scores = model.decision_function(resampled)
        predictions = model.predict(resampled)

        resampled['anomaly'] = predictions
        resampled['anomaly_score'] = anomaly_scores

        num_anomalies = sum(predictions == -1)
        logger.info(f"Nœud {node_name}: {num_anomalies} anomalies détectées sur {resampled.shape[0]} observations ({(num_anomalies / resampled.shape[0]) * 100:.2f}%)")
        return node_name, resampled.reset_index()

    except Exception as e:
        logger.error(f"Erreur lors du traitement du nœud {node_name}: {str(e)}")
        return node_name, None

def detect_anomalies(df, node_column, time_column, use_parallel=True, max_workers=None, batch_size=100):
    models = {}
    anomalies_results = []

    df_processed = preprocess_data(df, time_column)
    grouped = df_processed.groupby(node_column)
    unique_nodes = list(grouped.groups.keys())
    logger.info(f"Traitement de {len(unique_nodes)} nœuds uniques.")

    if max_workers is None:
        max_workers = min(4, len(unique_nodes)) if len(unique_nodes) > 0 else 1

    node_tuples = [(node, grouped.get_group(node)) for node in unique_nodes]
    batches = [node_tuples[i:i+batch_size] for i in range(0, len(node_tuples), batch_size)]
    logger.info(f"Traitement des nœuds en {len(batches)} batchs de maximum {batch_size} nœuds chacun.")

    if use_parallel:
        for batch in tqdm(batches, desc="Traitement des batchs de nœuds"):
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(process_node, nt): nt[0] for nt in batch}
                for future in as_completed(futures):
                    node = futures[future]
                    try:
                        node_name, result = future.result()
                        if result is not None:
                            anomalies_results.append(result)
                        else:
                            logger.info(f"Nœud {node_name} ignoré durant le traitement.")
                    except Exception as e:
                        logger.error(f"Erreur lors du traitement parallèle du nœud {node}: {str(e)}")
    else:
        for batch in tqdm(batches, desc="Traitement des batchs de nœuds"):
            for nt in batch:
                node_name, result = process_node(nt)
                if result is not None:
                    anomalies_results.append(result)

    if anomalies_results:
        result_df = pd.concat(anomalies_results, ignore_index=True)
    else:
        result_df = pd.DataFrame()

    logger.info(f"Détection d'anomalies terminée. {len(result_df)} observations avec anomalies détectées.")
    return models, result_df

def analyze_anomalies(result_df, node_column=None):
    if result_df.empty:
        logger.warning("Aucune anomalie détectée ou résultats vides.")
        return None

    total_observations = len(result_df)
    total_anomalies = (result_df['anomaly'] == -1).sum()
    anomaly_percentage = (total_anomalies / total_observations) * 100

    summary = {
        'total_observations': total_observations,
        'total_anomalies': total_anomalies,
        'anomaly_percentage': anomaly_percentage
    }

    if node_column and node_column in result_df.columns:
        node_stats = result_df.groupby(node_column)['anomaly'].apply(
            lambda x: (x == -1).sum() / len(x) * 100
        ).sort_values(ascending=False)
        summary['node_anomaly_percentages'] = node_stats

    return summary

if __name__ == "__main__":
    try:
        models_peag, results_peag = detect_anomalies(df, 'peag_nro', 'date_hour', use_parallel=True, batch_size=100)
        models_olt, results_olt = detect_anomalies(df, 'olt_name', 'date_hour', use_parallel=True, batch_size=100)
        models_boucle, results_boucle = detect_anomalies(df, 'boucle', 'date_hour', use_parallel=True, batch_size=100)

        peag_analysis = analyze_anomalies(results_peag, 'peag_nro')
        if not results_peag.empty:
            print("\nAperçu des résultats PEAG:")
            print(results_peag.head())
        if peag_analysis:
            print("\nAnalyse des anomalies PEAG:")
            for key, value in peag_analysis.items():
                if key != 'node_anomaly_percentages':
                    print(f"{key}: {value}")
            if 'node_anomaly_percentages' in peag_analysis:
                print("\nPourcentage d'anomalies par nœud (top 5) PEAG:")
                print(peag_analysis['node_anomaly_percentages'].head(5))

        olt_analysis = analyze_anomalies(results_olt, 'olt_name')
        if not results_olt.empty:
            print("\nAperçu des résultats OLT:")
            print(results_olt.head())
        if olt_analysis:
            print("\nAnalyse des anomalies OLT:")
            for key, value in olt_analysis.items():
                if key != 'node_anomaly_percentages':
                    print(f"{key}: {value}")
            if 'node_anomaly_percentages' in olt_analysis:
                print("\nPourcentage d'anomalies par nœud (top 5) OLT:")
                print(olt_analysis['node_anomaly_percentages'].head(5))

        boucle_analysis = analyze_anomalies(results_boucle, 'boucle')
        if not results_boucle.empty:
            print("\nAperçu des résultats BOUCLE:")
            print(results_boucle.head())
        if boucle_analysis:
            print("\nAnalyse des anomalies BOUCLE:")
            for key, value in boucle_analysis.items():
                if key != 'node_anomaly_percentages':
                    print(f"{key}: {value}")
            if 'node_anomaly_percentages' in boucle_analysis:
                print("\nPourcentage d'anomalies par nœud (top 5) BOUCLE:")
                print(boucle_analysis['node_anomaly_percentages'].head(5))

    except Exception as e:
        logger.error(f"Erreur lors de l'exécution principale: {str(e)}")
