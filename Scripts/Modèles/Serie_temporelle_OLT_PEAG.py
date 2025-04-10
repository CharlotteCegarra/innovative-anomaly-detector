# -----------------------------------------------------------------------------
# DÉTECTION D'ANOMALIES PAR MOYENNE MOBILE (ROLLING WINDOW)
# -----------------------------------------------------------------------------
# * Objectif :
#     Identifier les points anormaux dans la latence DNS moyenne (pondérée)
#     pour chaque nœud `olt_peag`, en utilisant un filtre basé sur la moyenne
#     mobile et l’écart-type.

# * Sortie :
#     - Courbe temporelle par nœud avec mise en évidence des pics anormaux
# -----------------------------------------------------------------------------


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



grouped['date_hour'] = pd.to_datetime(grouped['date_hour'])

rolling_window = 5       
anomaly_threshold = 2.0   

for group, grp_data in grouped.groupby('olt_peag'):
    grp_data = grp_data.sort_values('date_hour').reset_index(drop=True)

    rolling_mean = grp_data['weighted_avg_dns_time'].rolling(window=rolling_window, min_periods=1).mean()
    rolling_std  = grp_data['weighted_avg_dns_time'].rolling(window=rolling_window, min_periods=1).std()

    anomalies = grp_data[np.abs(grp_data['weighted_avg_dns_time'] - rolling_mean) > anomaly_threshold * rolling_std]

    plt.figure(figsize=(12, 6))
    plt.plot(grp_data['date_hour'], grp_data['weighted_avg_dns_time'], marker='o', linestyle='-', label='Weighted Avg DNS Time')
    plt.plot(grp_data['date_hour'], rolling_mean, color='orange', linestyle='--', linewidth=2, label='Moyenne mobile')

    if not anomalies.empty:
        plt.scatter(anomalies['date_hour'], anomalies['weighted_avg_dns_time'], color='red', s=100, zorder=5, label='Anomalies')

    plt.title(f"Série temporelle et détection d'anomalies pour {group}")
    plt.xlabel("Date et Heure")
    plt.ylabel("Weighted Avg DNS Time")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
