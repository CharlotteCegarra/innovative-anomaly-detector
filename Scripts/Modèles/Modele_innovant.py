# -----------------------------------------------------------------------------
# PIPELINE DÉTECTION D’ANOMALIES PAR STL, RECONSTRUCTION DE GRAPHES ET CLASSIFICATION
# -----------------------------------------------------------------------------
# * Brique 1 – STL Multirésolution (STiLe)
#     - Application de la décomposition STL sur la variable `weighted_avg_dns_time`
#     - Analyse multirésolution (1h, 6h, 24h) pour détecter des anomalies temporelles
#     - Résultats stockés par nœud (`olt_peag`) et visualisation des anomalies détectées
#     - Identification d’anomalies "uniques" détectées uniquement à une résolution fine (1h)

# * Brique 2 – Reconstruction du réseau sous forme de graphe
#     - Construction d’un DataFrame agrégé par nœud contenant les scores d’anomalie par résolution
#     - Création d’un graphe NetworkX où chaque nœud est un `olt_peag`, enrichi d’attributs :
#         - Criticité (moyenne des scores d’anomalie)
#         - Nombre de clients et variabilité temporelle
#     - Connexions créées entre nœuds partageant un préfixe OLT
#     - Visualisation statique du graphe avec criticité en couleur et taille proportionnelle aux clients

# * Brique 3 – Classification interprétable avec SHAP
#     - Modélisation supervisée d’instabilité à partir des scores d’anomalie (features)
#     - Entraînement d’un RandomForest pour prédire si un nœud est instable
#     - Interprétation du modèle avec SHAP pour identifier les variables les plus influentes
#     - Visualisation des contributions SHAP pour un échantillon test
# -----------------------------------------------------------------------------


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import logging
from statsmodels.tsa.seasonal import STL
from tqdm import tqdm
from pyvis.network import Network
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import shap



# Brique 1 – STL Multirésolution (STiLe) avec weighted_avg_dns_time, par batch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

grouped['date_hour'] = pd.to_datetime(grouped['date_hour'], errors='coerce')

def detect_anomalies_stl(series, period, threshold=2.0):
    stl = STL(series, period=period, robust=True)
    res = stl.fit()
    resid = res.resid
    std_resid = np.std(resid)
    anomalies = np.abs(resid) > threshold * std_resid
    return res, anomalies

resolutions = {
    '1h': {'freq': '1h', 'period': 24, 'color': 'red'},
    '6h': {'freq': '6h', 'period': 4, 'color': 'orange'},
    '24h': {'freq': '24h', 'period': 7, 'color': 'yellow'}
}

node_tuples = list(grouped.groupby('olt_peag'))
batch_size = 100
batches = [node_tuples[i:i+batch_size] for i in range(0, len(node_tuples), batch_size)]
logger.info(f"Traitement de {len(node_tuples)} nœuds en {len(batches)} batchs de taille maximum {batch_size}.")

results_stl = {}

for batch in tqdm(batches, desc="Traitement des batchs de nœuds"):
    for node, grp in batch:
        if 'olt_peag' in grp.columns:
            grp = grp.drop(columns=['olt_peag'])
        node_results = {}
        for res_key, params in resolutions.items():
            ts_resampled = grp.set_index('date_hour').select_dtypes(include=[np.number]).resample(params['freq']).mean()
            if len(ts_resampled) < params['period']:
                continue
            try:
                stl_res, anomalies = detect_anomalies_stl(ts_resampled['weighted_avg_dns_time'],
                                                          period=params['period'], threshold=2.0)
            except Exception as e:
                logger.error(f"Erreur STL pour {node} à la résolution {res_key} : {e}")
                continue
            ts_resampled = ts_resampled.reset_index()
            ts_resampled['anomaly'] = anomalies.values
            node_results[res_key] = ts_resampled
        results_stl[node] = node_results

for node, res_dict in results_stl.items():
    plt.figure(figsize=(14, 6))
    for res_key, ts_df in res_dict.items():
        if ts_df.empty:
            continue
        plt.plot(ts_df['date_hour'], ts_df['weighted_avg_dns_time'], marker='o', linestyle='-',
                 color=resolutions[res_key]['color'], label=f"{res_key} weighted_avg_dns_time")
        anomalies = ts_df[ts_df['anomaly']]
        plt.scatter(anomalies['date_hour'], anomalies['weighted_avg_dns_time'],
                    color='black', s=80, label=f"{res_key} anomaly" if not anomalies.empty else "")
    if '1h' in res_dict:
        anomalies_1h = res_dict['1h'][res_dict['1h']['anomaly']]
        for _, row in anomalies_1h.iterrows():
            ts = row['date_hour']
            exclusive = True
            for other in ['6h', '24h']:
                if other in res_dict:
                    if ts in res_dict[other]['date_hour'].values:
                        idx = res_dict[other]['date_hour'] == ts
                        if res_dict[other][idx]['anomaly'].any():
                            exclusive = False
            if exclusive:
                plt.annotate("Anomalie unique", xy=(ts, row['weighted_avg_dns_time']),
                             xytext=(ts, row['weighted_avg_dns_time'] + 1),
                             arrowprops=dict(facecolor='black', shrink=0.05))
    plt.title(f"Comparaison multirésolution des anomalies pour {node}")
    plt.xlabel("Date et Heure")
    plt.ylabel("Weighted Avg DNS Time")
    plt.legend()
    plt.tight_layout()
    plt.show()





#Brique 2 – Reconstruction du réseau sous forme de graphe

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pyvis.network import Network
import pickle

nodes_list = []
for node, res_dict in results_stl.items():
    node_dict = {'olt_peag': node}
    for res_key in resolutions.keys():
        if res_key in res_dict:
            anomaly_rate = res_dict[res_key]['anomaly'].mean()
            node_dict[f'anomaly_score_{res_key}'] = anomaly_rate
        else:
            node_dict[f'anomaly_score_{res_key}'] = np.nan
    if 'nb_client_total' in grouped.columns:
        node_dict['num_clients'] = grouped[grouped['olt_peag'] == node]['nb_client_total'].mean()
    else:
        node_dict['num_clients'] = np.nan
    node_dict['std_value'] = grouped[grouped['olt_peag'] == node]['weighted_avg_dns_time'].std()
    nodes_list.append(node_dict)

df_nodes = pd.DataFrame(nodes_list)

G = nx.Graph()
for idx, row in df_nodes.iterrows():
    scores = [row[f'anomaly_score_{res}'] for res in resolutions.keys() if not pd.isnull(row[f'anomaly_score_{res}'])]
    criticite = np.mean(scores) if scores else 0
    G.add_node(row['olt_peag'], criticite=criticite, num_clients=row.get('num_clients', np.nan), std_value=row.get('std_value', np.nan))

nodes = list(G.nodes())
for i in range(len(nodes)):
    for j in range(i + 1, len(nodes)):
        if nodes[i].split('_')[0] == nodes[j].split('_')[0]:
            G.add_edge(nodes[i], nodes[j])

plt.figure(figsize=(8, 8))
pos = nx.spring_layout(G, seed=42)
node_sizes = [G.nodes[n]['num_clients'] * 100 if pd.notnull(G.nodes[n]['num_clients']) else 300 for n in G.nodes()]
node_colors = [G.nodes[n]['criticite'] for n in G.nodes()]
nx.draw(G, pos, with_labels=True, node_size=node_sizes, node_color=node_colors, cmap=plt.cm.Reds, edge_color='gray')
plt.title("Graphe réseau des olt_peag")
plt.show()





# Brique 3 - Classification interprétable avec SHAP sur df_results_stl


features_stl = df_results_stl[['anomaly_score_1h', 'anomaly_score_6h', 'anomaly_score_24h']].copy()
features_stl = features_stl.fillna(features_stl.median())
target_stl = (df_results_stl['anomaly_score_1h'] > 0.5).astype(int)

X_train_stl, X_test_stl, y_train_stl, y_test_stl = train_test_split(features_stl, target_stl, test_size=0.3, random_state=42)

rf_model_stl = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_stl.fit(X_train_stl, y_train_stl)

explainer_stl = shap.TreeExplainer(rf_model_stl)
shap_values_stl = explainer_stl.shap_values(X_test_stl)

shap.initjs()
sample_index_stl = 0

if isinstance(shap_values_stl, np.ndarray):
    shap.bar_plot(shap_values_stl[sample_index_stl], feature_names=X_test_stl.columns, max_display=5)
elif isinstance(shap_values_stl, list):
    shap.bar_plot(shap_values_stl[1][sample_index_stl], feature_names=X_test_stl.columns, max_display=5)

plt.title("Top 5 variables influentes - SHAP bar plot (STL results)")
plt.show()
