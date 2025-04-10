# -----------------------------------------------------------------------------
# SCRIPT D'EXPLORATION ET DE PRÉPARATION DES DONNÉES POUR ANALYSE DE LA LATENCE
# -----------------------------------------------------------------------------
# - Lecture et inspection des données issues d’un fichier Parquet
# - Description statistique et exploration des variables numériques et catégorielles
# - Analyse des valeurs manquantes (globales et conditionnelles)
# - Visualisations univariées (histogrammes, boxplots, KDE) pour les métriques clés
# - Visualisations multivariées : effets horaires, heatmap temporelle, corrélation
# - Analyse temporelle par heure et par jour : agrégations, détection d'anomalies (seuils dynamiques, Shewhart)
# - Analyse géographique : latence par département avec cartes interactives (folium) et barplots
# - Comparaisons par équipements (OLT, NRO) : barplots et distributions
# - Prétraitement des données : imputation des valeurs manquantes
# - Feature engineering pondéré :
#       - Moyennes et variances pondérées par le nombre de tests
#       - Agrégation par nœud (`olt_peag`) et horodatage
# - Construction d’un DataFrame final `grouped` avec variables pondérées prêtes pour la modélisation
# -----------------------------------------------------------------------------


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
import geopandas as gpd
import folium
from IPython.display import display
from tqdm import tqdm


#Importation des données
df = pd.read_parquet("250327_tests_fixe_dns_sah_202412_202501.parquet")

#Lecture des données
print("Information du DataFrame :\n", df.info(), "\n")
print("Aperçu du DataFrame :\n", df.head(), "\n")
print("Type des variables :\n", df.dtypes, "\n")

# Listes de variables
numerical_variables = [
    "nb_test_dns", "avg_dns_time", "std_dns_time",
    "nb_test_scoring", "avg_latence_scoring", "std_latence_scoring",
    "avg_score_scoring", "std_score_scoring", "nb_client_total"
]

categorical_variables = [
    "date_hour", "code_departement", "olt_model", "olt_name",
    "peag_nro", "boucle", "dsp", "pebib", "pop_dns"
]

#Description des variables numériques 
print(df[numerical_variables].describe())

# Nombre de catégories uniques pour chaque variable catégorielle
print(df[categorical_variables].nunique()) 

# Valeurs manquantes
missing_values = df.isnull().sum()
print("Valeurs manquantes par colonne :\n", missing_values[missing_values > 0])

for cat in categorical_variables:
    print(f"\nPour la variable catégorielle '{cat}':")
    missing_cat_df = df[df[cat].isnull()]

    print("Valeurs manquantes dans les variables numériques pour les lignes où", cat, "est manquant :")
    for num in numerical_variables:
        missing_count = missing_cat_df[num].isnull().sum()
        print(f"{num:25s} {missing_count}")

    all_numeric_missing = missing_cat_df[numerical_variables].isnull().all(axis=1)
    print("\nNombre de lignes où toutes les variables numériques sont manquantes :", all_numeric_missing.value_counts())

#Visualisations
## Echantillon aléatoire de 10 000 lignes pour réduire la charge mémoire
df_sample = df.sample(n=50000, random_state=42)

## Heatmap des valeurs manquantes
plt.figure(figsize=(12, 6))
sns.heatmap(df_sample.isnull(), cbar=False, cmap='viridis')
plt.title("Heatmap des valeurs manquantes (échantillon)")
plt.show()

## Histogrammes des variables numériques
df_numeric_sample = df_sample.select_dtypes(include=['float64', 'int64'])
df_numeric_sample.hist(figsize=(12, 8), bins=50, color='blue', edgecolor='black')
plt.suptitle("Histogramme des variables numériques (échantillon)")
plt.show()

## Boxplots pour détecter les valeurs aberrantes
plt.figure(figsize=(12, 8))
sns.boxplot(data=df_numeric_sample)
plt.xticks(rotation=90)
plt.title("Boxplot des variables numériques (échantillon)")
plt.show()

## Analyse Univariée

numerical_variables = [
    "avg_dns_time", "std_dns_time", "avg_latence_scoring",
    "std_latence_scoring", "avg_score_scoring", "std_score_scoring",
    "nb_client_total"
]

### Histogrammes + KDE plots 
for var in numerical_variables:
    plt.figure(figsize=(10, 5))
    sns.histplot(df_sample[var], kde=True, bins=50, color='blue')
    plt.title(f"Distribution de {var}")
    plt.xlabel(var)
    plt.ylabel("Fréquence")
    plt.show()

### Boxplots pour détecter les valeurs extrêmes sur un échantillon
plt.figure(figsize=(12, 8))
sns.boxplot(data=df_sample[numerical_variables])
plt.xticks(rotation=90)
plt.title("Boxplot des variables numériques (Détection des valeurs extrêmes)")
plt.show()

### Répartition des équipements (olt_model, olt_name, peag_nro) - On affiche uniquement les 20 plus fréquents
categorical_variables = ["olt_model", "olt_name", "peag_nro"]

for cat_var in categorical_variables:
    plt.figure(figsize=(12, 5))
    top_values = df_sample[cat_var].value_counts().index[:20] 
    sns.countplot(y=df_sample[cat_var], order=top_values)
    plt.title(f"Répartition des équipements : {cat_var}")
    plt.xlabel("Nombre d'occurrences")
    plt.ylabel(cat_var)
    plt.show()


fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))

# Histogramme + KDE pour nb_test_dns
data = df['nb_test_dns'].dropna()
mean_data = data.mean()
axes[0, 0].hist(data, bins=30, density=True, color='blue', alpha=0.6, edgecolor='black')
kde = gaussian_kde(data)
x = np.linspace(data.min(), data.max(), 1000)
axes[0, 0].plot(x, kde(x), color='red', linewidth=2, label=f'Moyenne: {mean_data:.2f}')
axes[0, 0].set_title("Histogramme + KDE de nb_test_dns")
axes[0, 0].set_xlabel("Nombre de tests DNS")
axes[0, 0].set_ylabel("Densité")
axes[0, 0].legend()

# Histogramme + KDE pour nb_test_scoring
data = df['nb_test_scoring'].dropna()
mean_data = data.mean()
axes[0, 1].hist(data, bins=30, density=True, color='green', alpha=0.6, edgecolor='black')
kde = gaussian_kde(data)
x = np.linspace(data.min(), data.max(), 1000)
axes[0, 1].plot(x, kde(x), color='red', linewidth=2, label=f'Moyenne: {mean_data:.2f}')
axes[0, 1].set_title("Histogramme + KDE de nb_test_scoring")
axes[0, 1].set_xlabel("Nombre de tests de scoring")
axes[0, 1].set_ylabel("Densité")
axes[0, 1].legend()

# Histogramme + KDE pour avg_dns_time
data = df['avg_dns_time'].dropna()
mean_data = data.mean()
axes[0, 2].hist(data, bins=30, density=True, color='orange', alpha=0.6, edgecolor='black')
kde = gaussian_kde(data)
x = np.linspace(data.min(), data.max(), 1000)
axes[0, 2].plot(x, kde(x), color='red', linewidth=2, label=f'Moyenne: {mean_data:.2f}')
axes[0, 2].set_title("Histogramme + KDE de avg_dns_time")
axes[0, 2].set_xlabel("Latence DNS moyenne")
axes[0, 2].set_ylabel("Densité")
axes[0, 2].legend()

# Histogramme + KDE pour avg_latence_scoring
data = df['avg_latence_scoring'].dropna()
mean_data = data.mean()
axes[1, 0].hist(data, bins=30, density=True, color='red', alpha=0.6, edgecolor='black')
kde = gaussian_kde(data)
x = np.linspace(data.min(), data.max(), 1000)
axes[1, 0].plot(x, kde(x), color='blue', linewidth=2, label=f'Moyenne: {mean_data:.2f}')
axes[1, 0].set_title("Histogramme + KDE de avg_latence_scoring")
axes[1, 0].set_xlabel("Latence de scoring moyenne")
axes[1, 0].set_ylabel("Densité")
axes[1, 0].legend()

# Histogramme + KDE pour avg_score_scoring
data = df['avg_score_scoring'].dropna()
mean_data = data.mean()
axes[1, 1].hist(data, bins=30, density=True, color='purple', alpha=0.6, edgecolor='black')
kde = gaussian_kde(data)
x = np.linspace(data.min(), data.max(), 1000)
axes[1, 1].plot(x, kde(x), color='red', linewidth=2, label=f'Moyenne: {mean_data:.2f}')
axes[1, 1].set_title("Histogramme + KDE de avg_score_scoring")
axes[1, 1].set_xlabel("Score moyen de scoring")
axes[1, 1].set_ylabel("Densité")
axes[1, 1].legend()
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

## Histogramme + KDE pour nb_test_dns
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))

data = df['nb_test_dns'].dropna()
mean_data = data.mean()
axes[0, 0].hist(data, bins=30, density=True, color='blue', alpha=0.6, edgecolor='black')
kde = gaussian_kde(data)
x = np.linspace(data.min(), data.max(), 1000)
axes[0, 0].plot(x, kde(x), color='red', linewidth=2, label=f'Moyenne: {mean_data:.2f}')
axes[0, 0].set_title("Histogramme + KDE de nb_test_dns")
axes[0, 0].set_xlabel("Nombre de tests DNS")
axes[0, 0].set_ylabel("Densité")
axes[0, 0].legend()

## Histogramme + KDE pour nb_test_scoring
data = df['nb_test_scoring'].dropna()
mean_data = data.mean()
axes[0, 1].hist(data, bins=30, density=True, color='green', alpha=0.6, edgecolor='black')
kde = gaussian_kde(data)
x = np.linspace(data.min(), data.max(), 1000)
axes[0, 1].plot(x, kde(x), color='red', linewidth=2, label=f'Moyenne: {mean_data:.2f}')
axes[0, 1].set_title("Histogramme + KDE de nb_test_scoring")
axes[0, 1].set_xlabel("Nombre de tests de scoring")
axes[0, 1].set_ylabel("Densité")
axes[0, 1].legend()

## Histogramme + KDE pour avg_dns_time
data = df['avg_dns_time'].dropna()
mean_data = data.mean()
axes[0, 2].hist(data, bins=30, density=True, color='orange', alpha=0.6, edgecolor='black')
kde = gaussian_kde(data)
x = np.linspace(data.min(), data.max(), 1000)
axes[0, 2].plot(x, kde(x), color='red', linewidth=2, label=f'Moyenne: {mean_data:.2f}')
axes[0, 2].set_title("Histogramme + KDE de avg_dns_time")
axes[0, 2].set_xlabel("Latence DNS moyenne")
axes[0, 2].set_ylabel("Densité")
axes[0, 2].legend()

## Histogramme + KDE pour avg_latence_scoring
data = df['avg_latence_scoring'].dropna()
mean_data = data.mean()
axes[1, 0].hist(data, bins=30, density=True, color='red', alpha=0.6, edgecolor='black')
kde = gaussian_kde(data)
x = np.linspace(data.min(), data.max(), 1000)
axes[1, 0].plot(x, kde(x), color='blue', linewidth=2, label=f'Moyenne: {mean_data:.2f}')
axes[1, 0].set_title("Histogramme + KDE de avg_latence_scoring")
axes[1, 0].set_xlabel("Latence de scoring moyenne")
axes[1, 0].set_ylabel("Densité")
axes[1, 0].legend()

## Histogramme + KDE pour avg_score_scoring
data = df['avg_score_scoring'].dropna()
mean_data = data.mean()
axes[1, 1].hist(data, bins=30, density=True, color='purple', alpha=0.6, edgecolor='black')
kde = gaussian_kde(data)
x = np.linspace(data.min(), data.max(), 1000)
axes[1, 1].plot(x, kde(x), color='red', linewidth=2, label=f'Moyenne: {mean_data:.2f}')
axes[1, 1].set_title("Histogramme + KDE de avg_score_scoring")
axes[1, 1].set_xlabel("Score moyen de scoring")
axes[1, 1].set_ylabel("Densité")
axes[1, 1].legend()

axes[1, 2].axis('off')
plt.tight_layout()
plt.show()

## Création de courbes de densité (KDE) pour visualiser la distribution des variables numériques
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))

### Densité de nb_test_dns
df['nb_test_dns'].dropna().plot.kde(ax=axes[0, 0], color='blue')
axes[0, 0].set_title("Densité de nb_test_dns")
axes[0, 0].set_xlabel("Nombre de tests DNS")
axes[0, 0].set_ylabel("Densité")

### Densité de nb_test_scoring
df['nb_test_scoring'].dropna().plot.kde(ax=axes[0, 1], color='green')
axes[0, 1].set_title("Densité de nb_test_scoring")
axes[0, 1].set_xlabel("Nombre de tests de scoring")
axes[0, 1].set_ylabel("Densité")

### Densité de avg_dns_time
df['avg_dns_time'].dropna().plot.kde(ax=axes[0, 2], color='orange')
axes[0, 2].set_title("Densité de avg_dns_time")
axes[0, 2].set_xlabel("Latence DNS moyenne")
axes[0, 2].set_ylabel("Densité")

### Densité de avg_latence_scoring
df['avg_latence_scoring'].dropna().plot.kde(ax=axes[1, 0], color='red')
axes[1, 0].set_title("Densité de avg_latence_scoring")
axes[1, 0].set_xlabel("Latence de scoring moyenne")
axes[1, 0].set_ylabel("Densité")

### Densité de avg_score_scoring
df['avg_score_scoring'].dropna().plot.kde(ax=axes[1, 1], color='purple')
axes[1, 1].set_title("Densité de avg_score_scoring")
axes[1, 1].set_xlabel("Score moyen de scoring")
axes[1, 1].set_ylabel("Densité")

axes[1, 2].axis('off')
plt.tight_layout()
plt.show()

# Analyse multivariée 

df['hour'] = df['date_hour'].dt.hour
latency_dns = df.groupby('hour')['avg_dns_time'].mean()
latency_scoring = df.groupby('hour')['avg_latence_scoring'].mean()
latency_debit = df.groupby('hour')['avg_score_scoring'].mean()
plt.figure(figsize=(12, 6))
plt.plot(latency_dns.index, latency_dns.values, marker='o', linestyle='-', label='Latence DNS moyenne', color='orange')
plt.plot(latency_scoring.index, latency_scoring.values, marker='o', linestyle='-', label='Latence Scoring moyenne', color='red')
plt.plot(latency_debit.index, latency_debit.values, marker='o', linestyle='-', label='Latence Debit - KPIs moyenne', color='green')
plt.xlabel("Heure de la journée")
plt.ylabel("Latence moyenne")
plt.title("Impact horaire sur la latence")
plt.xticks(range(0, 24))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

## Boxplot pour avg_dns_time en fonction de l'heure
df.boxplot(column='avg_dns_time', by='hour', ax=axes[0], grid=False)
axes[0].set_title("Latence DNS par heure")
axes[0].set_xlabel("Heure")
axes[0].set_ylabel("Latence DNS")

## Boxplot pour avg_latence_scoring en fonction de l'heure
df.boxplot(column='avg_latence_scoring', by='hour', ax=axes[1], grid=False)
axes[1].set_title("Latence de scoring par heure")
axes[1].set_xlabel("Heure")
axes[1].set_ylabel("Latence de scoring")

## Boxplot pour avg_score_scoring en fonction de l'heure
df.boxplot(column='avg_score_scoring', by='hour', ax=axes[2], grid=False)
axes[2].set_title("Latence de débit - KPIs par heure")
axes[2].set_xlabel("Heure")
axes[2].set_ylabel("Score de scoring")

plt.suptitle("")
plt.tight_layout()
plt.show()

## Analyse temporelle 

df.sort_values(by='date_hour', inplace=True)
df_daily = df.set_index('date_hour').resample('D').agg({
    'nb_test_dns': 'sum',            
    'avg_dns_time': 'mean',          
})

### Détection des anomalies
dns_mean = df_daily['nb_test_dns'].mean()
dns_std = df_daily['nb_test_dns'].std()
dns_threshold = dns_mean + 2 * dns_std

lat_mean = df_daily['avg_dns_time'].mean()
lat_std = df_daily['avg_dns_time'].std()
lat_threshold = lat_mean + 2 * lat_std

df_daily['dns_anomaly'] = df_daily['nb_test_dns'] > dns_threshold
df_daily['lat_anomaly'] = df_daily['avg_dns_time'] > lat_threshold

### Nombre de test par jour 
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True)

axes[0].plot(df_daily.index, df_daily['nb_test_dns'], label='Nombre de tests DNS', color='blue')
anomalies_dns = df_daily[df_daily['dns_anomaly']]
axes[0].scatter(anomalies_dns.index, anomalies_dns['nb_test_dns'], color='red', label='Anomalies DNS')

axes[0].set_title("Nombre de tests DNS par jour (agrégation)")
axes[0].set_ylabel("Nombre de tests DNS")
axes[0].legend()

### Latence DNS moyenne par jour
axes[1].plot(df_daily.index, df_daily['avg_dns_time'], label='Latence DNS moyenne', color='green')
anomalies_lat = df_daily[df_daily['lat_anomaly']]
axes[1].scatter(anomalies_lat.index, anomalies_lat['avg_dns_time'], color='orange', label='Anomalies latence')

axes[1].set_title("Latence DNS moyenne par jour (agrégation)")
axes[1].set_ylabel("Latence DNS moyenne (ms ou s)")
axes[1].set_xlabel("Date")
axes[1].legend()

plt.tight_layout()
plt.show()

### Détection des anomalies par seuil
df['date_hour'] = pd.to_datetime(df['date_hour'], format='%Y-%m-%d %H:%M:%S')
df.sort_values(by='date_hour', inplace=True)

df_hourly = df.set_index('date_hour').resample('H').agg({
    'nb_test_dns': 'sum',       # Somme des tests DNS par heure
    'avg_dns_time': 'mean'      # Moyenne de la latence DNS par heure
})

#### Pour le nombre de tests DNS
dns_mean = df_hourly['nb_test_dns'].mean()
dns_std = df_hourly['nb_test_dns'].std()
dns_threshold = dns_mean + 2 * dns_std

df_hourly['dns_anomaly'] = df_hourly['nb_test_dns'] > dns_threshold

#### Pour la latence DNS
lat_mean = df_hourly['avg_dns_time'].mean()
lat_std = df_hourly['avg_dns_time'].std()
lat_threshold = lat_mean + 2 * lat_std

df_hourly['lat_anomaly'] = df_hourly['avg_dns_time'] > lat_threshold

## Calcul des limites de contrôle
### Pour nb_test_dns
metric1 = 'nb_test_dns'
mean_val1 = df_hourly[metric1].mean()
std_val1 = df_hourly[metric1].std()
ucl1 = mean_val1 + 3 * std_val1
lcl1 = mean_val1 - 3 * std_val1

### Pour avg_dns_time
metric2 = 'avg_dns_time'
mean_val2 = df_hourly[metric2].mean()
std_val2 = df_hourly[metric2].std()
ucl2 = mean_val2 + 3 * std_val2
lcl2 = mean_val2 - 3 * std_val2

### Représentation graphique
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(14, 10), sharex=True)

ax1.plot(df_hourly.index, df_hourly[metric1], marker='o', linestyle='-', label=metric1, color='blue')
ax1.axhline(mean_val1, color='green', linestyle='--', label='Moyenne')

ax1.axhline(ucl1, color='red', linestyle='--', label='UCL (Moyenne + 3σ)')
ax1.axhline(lcl1, color='red', linestyle='--', label='LCL (Moyenne - 3σ)')

anomalies1 = df_hourly[(df_hourly[metric1] > ucl1) | (df_hourly[metric1] < lcl1)]
ax1.scatter(anomalies1.index, anomalies1[metric1], color='red', zorder=5, label='Anomalies')

ax1.set_title(f'Control Chart (Shewhart) pour {metric1}')
ax1.set_ylabel(metric1)
ax1.legend()


ax2.plot(df_hourly.index, df_hourly[metric2], marker='o', linestyle='-', label=metric2, color='orange')
ax2.axhline(mean_val2, color='green', linestyle='--', label='Moyenne')

ax2.axhline(ucl2, color='red', linestyle='--', label='UCL (Moyenne + 3σ)')
ax2.axhline(lcl2, color='red', linestyle='--', label='LCL (Moyenne - 3σ)')

anomalies2 = df_hourly[(df_hourly[metric2] > ucl2) | (df_hourly[metric2] < lcl2)]
ax2.scatter(anomalies2.index, anomalies2[metric2], color='red', zorder=5, label='Anomalies')

ax2.set_title(f'Control Chart (Shewhart) pour {metric2}')
ax2.set_ylabel(metric2)
ax2.set_xlabel('Date et heure')
ax2.legend()

plt.tight_layout()
plt.show()

## Affichage du nombre d'observation par heure 

plt.figure(figsize=(10, 6)) 
plt.hist(df['hour'], bins=24, edgecolor='black', alpha=0.7)  
plt.title('Distribution des heures')
plt.xlabel('Heure')
plt.ylabel('Nombre d\'observations')
plt.xticks(range(0, 24))  
plt.grid(True)
plt.show()

## Evolution de la latence moyenne journalière

numeric_cols = new_df.select_dtypes(include=["number"]).columns

df_hourly = new_df[numeric_cols].resample("h").mean()
df_daily = new_df[numeric_cols].resample("D").mean()
df_weekly = new_df[numeric_cols].resample("W").mean()

plt.figure(figsize=(15, 6))
sns.lineplot(x=df_daily.index, y=df_daily["avg_latence_scoring"], label="Latence Moyenne")
plt.title("Évolution de la latence moyenne journalière")
plt.xlabel("Date")
plt.ylabel("Latence moyenne (ms)")
plt.xticks(rotation=45)
plt.legend()
plt.show()

## Distribution de la latence par heure de la journée
if "hour" not in new_df.columns:
    new_df["hour"] = new_df.index.hour  

plt.figure(figsize=(12, 6))
sns.boxplot(x=new_df["hour"], y=new_df["avg_latence_scoring"])
plt.title("Distribution de la latence par heure de la journée")
plt.xlabel("Heure")
plt.ylabel("Latence moyenne (ms)")
plt.show()

## Heatmap des variations temporelles : latence moyenne par jour et heure

new_df["day"] = new_df.index.date
pivot_table = new_df.pivot_table(values="avg_latence_scoring", index="hour", columns="day", aggfunc="mean")

plt.figure(figsize=(15, 6))
sns.heatmap(pivot_table, cmap="coolwarm", linewidths=0.5)
plt.title("Heatmap des variations de latence au fil du temps")
plt.xlabel("Jour")
plt.ylabel("Heure de la journée")
plt.show()

## Détection des pics latents

# Identification des périodes critiques
from scipy.signal import find_peaks

# Détection des pics de latence (seuil dynamique basé sur l'écart-type)
threshold = new_df["avg_latence_scoring"].mean() + 2 * new_df["avg_latence_scoring"].std()
peaks, _ = find_peaks(new_df["avg_latence_scoring"], height=threshold)

plt.figure(figsize=(15, 6))
sns.lineplot(x=new_df.index, y=new_df["avg_latence_scoring"], label="Latence Moyenne")
plt.scatter(new_df.index[peaks], new_df["avg_latence_scoring"].iloc[peaks], color='red', label="Pics détectés")
plt.axhline(threshold, color='gray', linestyle='--', label="Seuil des pics")
plt.title("Détection des pics de latence")
plt.xlabel("Date")
plt.ylabel("Latence moyenne (ms)")
plt.legend()
plt.xticks(rotation=45)
plt.show()

# Détection de la forte variabilité (écart-type sur une fenêtre glissante)

window_size = 10  
new_df["rolling_std"] = new_df["avg_latence_scoring"].rolling(window=window_size).std()

variability_threshold = new_df["rolling_std"].mean() + new_df["rolling_std"].std()
high_variability = new_df[new_df["rolling_std"] > variability_threshold]

plt.figure(figsize=(15, 6))
sns.lineplot(x=new_df.index, y=new_df["rolling_std"], label="Écart-type glissant")
plt.axhline(variability_threshold, color='red', linestyle='--', label="Seuil de variabilité")
plt.title("Détection des périodes de forte variabilité de latence")
plt.xlabel("Date")
plt.ylabel("Écart-type de la latence")
plt.legend()
plt.xticks(rotation=45)
plt.show()

## Analyse géographique

df["code_departement"] = df["code_departement"].astype(str)

url_geojson = "https://france-geojson.gregoiredavid.fr/repo/departements.geojson"
gdf_departements = gpd.read_file(url_geojson)

print("\nCodes départements dans GeoJSON:", gdf_departements["code"].head())

df_dept = df.groupby("code_departement", as_index=False)["avg_latence_scoring"].mean()
gdf = gdf_departements.merge(df_dept, left_on="code", right_on="code_departement", how="inner")
print("\nNombre de départements après fusion:", len(gdf))

m = folium.Map(location=[46.603354, 1.888334], zoom_start=5)

folium.Choropleth(
    geo_data=gdf,
    name="Latence par département",
    data=df_dept,
    columns=["code_departement", "avg_latence_scoring"],
    key_on="feature.properties.code",
    fill_color="YlOrRd",
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name="Latence moyenne (ms)"
).add_to(m)

for _, row in gdf.iterrows():
    folium.Marker(
        location=[row.geometry.centroid.y, row.geometry.centroid.x],
        popup=f"Dépt {row['code']}: {row['avg_latence_scoring']:.2f} ms",
        icon=folium.Icon(color="blue", icon="info-sign")
    ).add_to(m)

display(m)

## Latence moyenne par département 

plt.figure(figsize=(20, 8))  
df_dept_sorted = df_dept.sort_values("code_departement")

ax = sns.barplot(
    x="code_departement",
    y="avg_latence_scoring",
    data=df_dept_sorted,
    palette="coolwarm",
    order=df_dept_sorted["code_departement"]  # Forcer l'ordre explicite
)

plt.text(0.5, 0.95, f"Nombre de départements: {len(df_dept_sorted)}",
         horizontalalignment='center', transform=ax.transAxes)

plt.xticks(rotation=90, fontsize=8)
plt.title("Latence moyenne par département")
plt.xlabel("Code Département")
plt.ylabel("Latence Moyenne (ms)")
plt.tight_layout()
plt.show()


# Distribution des latences par département 
plt.figure(figsize=(20, 10))
sns.stripplot(x="code_departement", y="avg_latence_scoring", data=df,
              order=sorted(df["code_departement"].unique()),
              jitter=True, alpha=0.5)
plt.xticks(rotation=90, fontsize=8)
plt.title("Distribution des latences par département")
plt.xlabel("Code Département")
plt.ylabel("Latence (ms)")
plt.tight_layout()
plt.show()

## Latence moyenne des 20 équipements NRO les plus fréquents 

import folium
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display


# Conversion explicite en chaîne de caractères pour s'assurer que tous les codes sont du même type
df["code_departement"] = df["code_departement"].astype(str)

# 1. CARTE CHOROPLETH DES DÉPARTEMENTS
# -----------------------------------
# Charger les contours des départements français
url_geojson = "https://france-geojson.gregoiredavid.fr/repo/departements.geojson"
gdf_departements = gpd.read_file(url_geojson)

# Vérifier les codes dans les données GeoJSON
print("\nCodes départements dans GeoJSON:", gdf_departements["code"].head())

# Fusion avec la géométrie des départements
df_dept = df.groupby("code_departement", as_index=False)["avg_latence_scoring"].mean()
gdf = gdf_departements.merge(df_dept, left_on="code", right_on="code_departement", how="inner")
print("\nNombre de départements après fusion:", len(gdf))

# Création de la carte interactive
m = folium.Map(location=[46.603354, 1.888334], zoom_start=5)

# Ajouter un calque coloré pour la latence
folium.Choropleth(
    geo_data=gdf,
    name="Latence par département",
    data=df_dept,
    columns=["code_departement", "avg_latence_scoring"],
    key_on="feature.properties.code",
    fill_color="YlOrRd",
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name="Latence moyenne (ms)"
).add_to(m)

# Ajouter des marqueurs pour chaque département
for _, row in gdf.iterrows():
    folium.Marker(
        location=[row.geometry.centroid.y, row.geometry.centroid.x],
        popup=f"Dépt {row['code']}: {row['avg_latence_scoring']:.2f} ms",
        icon=folium.Icon(color="blue", icon="info-sign")
    ).add_to(m)

# Affichage de la carte
display(m)

# 2. BARPLOT DES LATENCES PAR DÉPARTEMENT
# --------------------------------------
# Affiche explicitement tous les départements
plt.figure(figsize=(20, 8))  # Augmenter la taille pour meilleure lisibilité

# Trier les données pour assurer l'ordre des départements
df_dept_sorted = df_dept.sort_values("code_departement")

# Tracer le graphique bar avec tous les départements explicitement
ax = sns.barplot(
    x="code_departement",
    y="avg_latence_scoring",
    data=df_dept_sorted,
    palette="coolwarm",
    order=df_dept_sorted["code_departement"]  # Forcer l'ordre explicite
)

# Afficher le nombre de départements sur le graphique
plt.text(0.5, 0.95, f"Nombre de départements: {len(df_dept_sorted)}",
         horizontalalignment='center', transform=ax.transAxes)

plt.xticks(rotation=90, fontsize=8)
plt.title("Latence moyenne par département")
plt.xlabel("Code Département")
plt.ylabel("Latence Moyenne (ms)")
plt.tight_layout()
plt.show()


# 3. DISTRIBUTION DES LATENCES PAR DÉPARTEMENT
# -----------------------------------------
# Afficher la distribution des latences pour chaque département
plt.figure(figsize=(20, 10))
sns.stripplot(x="code_departement", y="avg_latence_scoring", data=df,
              order=sorted(df["code_departement"].unique()),
              jitter=True, alpha=0.5)
plt.xticks(rotation=90, fontsize=8)
plt.title("Distribution des latences par département")
plt.xlabel("Code Département")
plt.ylabel("Latence (ms)")
plt.tight_layout()
plt.show()

# 4. COMPARAISON DES PERFORMANCES DES ÉQUIPEMENTS (OLT_MODEL, PEAG_NRO)
# --------------------------------------------------------------------

# 🔹 Barplot des latences par modèle OLT
df_olt = df.groupby("olt_model", as_index=False)["avg_latence_scoring"].mean()

plt.figure(figsize=(15, 6))
sns.barplot(x="olt_model", y="avg_latence_scoring", data=df_olt, palette="Set2",
            order=df_olt.sort_values("avg_latence_scoring")["olt_model"])
plt.xticks(rotation=45, fontsize=10)
plt.title("Latence moyenne par modèle OLT")
plt.xlabel("Modèle OLT")
plt.ylabel("Latence Moyenne (ms)")
plt.show()

top_nro = df["peag_nro"].value_counts().nlargest(20).index  # Prend les 20 plus fréquents
df_nro_filtered = df[df["peag_nro"].isin(top_nro)].groupby("peag_nro", as_index=False)["avg_latence_scoring"].mean()

plt.figure(figsize=(12, 8))  
sns.barplot(x="peag_nro", y="avg_latence_scoring", data=df_nro_filtered, palette="Set3",
            order=df_nro_filtered.sort_values("avg_latence_scoring")["peag_nro"])
plt.xticks(rotation=45, fontsize=10)
plt.title("Latence moyenne des 20 équipements NRO les plus fréquents")
plt.xlabel("Équipement NRO")
plt.ylabel("Latence Moyenne (ms)")
plt.show() 

plt.figure(figsize=(12, 8))  
sns.barplot(x="peag_nro", y="avg_latence_scoring", data=df_nro_filtered, palette="Set3",
            order=df_nro_filtered.sort_values("avg_latence_scoring")["peag_nro"])
plt.xticks(rotation=45, fontsize=10)
plt.title("Latence moyenne des 20 équipements NRO les plus fréquents")
plt.xlabel("Équipement NRO")
plt.ylabel("Latence Moyenne (ms)")
plt.show()

#Imputation des valeurs manquantes 

for col in numerical_variables:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

for col in categorical_variables:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])

# Création de variables 

df["olt_peag"] = df["olt_name"] + "_" + df["peag_nro"].astype(str)

n_chunks = 10
chunks = np.array_split(df, n_chunks)
agg_list = []

for chunk in tqdm(chunks, desc="Traitement des batchs"):

    chunk["weighted_dns_time"] = chunk["avg_dns_time"] * chunk["nb_test_dns"]
    chunk["weighted_dns_var"] = (chunk["std_dns_time"] ** 2) * chunk["nb_test_dns"]

    chunk["weighted_latence_scoring"] = chunk["avg_latence_scoring"] * chunk["nb_test_scoring"]
    chunk["weighted_latence_var"] = (chunk["std_latence_scoring"] ** 2) * chunk["nb_test_scoring"]

    chunk["weighted_score_scoring"] = chunk["avg_score_scoring"] * chunk["nb_test_scoring"]
    chunk["weighted_score_var"] = (chunk["std_score_scoring"] ** 2) * chunk["nb_test_scoring"]

    agg_chunk = chunk.groupby(["olt_peag", "date_hour"]).agg({
        "nb_test_dns": "sum",
        "weighted_dns_time": "sum",
        "weighted_dns_var": "sum",
        "nb_test_scoring": "sum",
        "weighted_latence_scoring": "sum",
        "weighted_latence_var": "sum",
        "weighted_score_scoring": "sum",
        "weighted_score_var": "sum",
        "nb_client_total": "sum"
    }).reset_index()

    agg_list.append(agg_chunk)

agg_all = pd.concat(agg_list)
grouped = agg_all.groupby(["olt_peag", "date_hour"]).agg({
    "nb_test_dns": "sum",
    "weighted_dns_time": "sum",
    "weighted_dns_var": "sum",
    "nb_test_scoring": "sum",
    "weighted_latence_scoring": "sum",
    "weighted_latence_var": "sum",
    "weighted_score_scoring": "sum",
    "weighted_score_var": "sum",
    "nb_client_total": "sum"
}).reset_index()
grouped["weighted_avg_dns_time"] = grouped["weighted_dns_time"] / grouped["nb_test_dns"]
grouped["weighted_std_dns_time"] = np.sqrt(grouped["weighted_dns_var"] / grouped["nb_test_dns"])

grouped["weighted_avg_latence_scoring"] = grouped["weighted_latence_scoring"] / grouped["nb_test_scoring"]
grouped["weighted_std_latence_scoring"] = np.sqrt(grouped["weighted_latence_var"] / grouped["nb_test_scoring"])

grouped["weighted_avg_score_scoring"] = grouped["weighted_score_scoring"] / grouped["nb_test_scoring"]
grouped["weighted_std_score_scoring"] = np.sqrt(grouped["weighted_score_var"] / grouped["nb_test_scoring"])

grouped["nb_test_dns"] = grouped["nb_test_dns"].round().astype("Int64", errors="ignore")
grouped["nb_test_scoring"] = grouped["nb_test_scoring"].round().astype("Int64", errors="ignore")
grouped["nb_client_total"] = grouped["nb_client_total"].round().astype("Int64", errors="ignore")

grouped = grouped.drop(columns=[
    "weighted_dns_time",
    "weighted_dns_var",
    "weighted_latence_scoring",
    "weighted_latence_var",
    "weighted_score_scoring",
    "weighted_score_var"
])

print(grouped.head())