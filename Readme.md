# Détection & Prédiction des Anomalies Réseau SFR

## Contexte & Problématique

Les box internet, bien qu’élémentaires pour l’accès à Internet, peuvent souffrir de dysfonctionnements non liés à l’appareil lui-même, mais à des problèmes situés plus loin dans l’infrastructure réseau (PM, PEAG, OLT...).  
Ces instabilités réseau sont souvent collectives, silencieuses, et complexes à localiser, affectant des milliers d’utilisateurs.

### Objectif du projet

Notre objectif est d’**anticiper** ces problèmes réseau avant qu’ils n’impactent les clients, en construisant une solution :
- basée sur les mesures de performance réseau (latence DNS, scoring…)
- capable de **détecter** les anomalies en temps réel
- et surtout de **prédire** les futures instabilités

## Solution développée

Pour répondre aux besoins métiers de SFR, nous avons conçu **NeuroGraph STiLe** (*Structural Temporal & Interpretable Learning Engine*), une solution innovante en trois briques :

### 1. STL multirésolution
Détection d’anomalies fines sur la latence (`weighted_avg_dns_time`) grâce à une décomposition **STL** appliquée à trois échelles temporelles :
- **1h** (granularité fine)
- **6h**
- **24h** (vue macro)
  
Chaque nœud `olt_peag` est analysé individuellement, et les anomalies sont comparées entre résolutions pour identifier des signaux faibles.

### 2. Graphe métier du réseau
Le réseau est modélisé comme un **graphe**, avec :
- **Nœuds** : `olt_peag`
- **Arêtes** : relations techniques (même préfixe OLT)
  
Chaque nœud est enrichi d’attributs métier : **criticité**, **nombre de clients**, **variabilité**. Cette structure facilite la visualisation des zones à risque.

### 3. Prédiction supervisée explicable
Passage au prédictif à l’aide de modèles supervisés :
- **Cible binaire** : instabilité définie par un **seuil dynamique (MAD)**
- **Modèle RandomForest** avec équilibrage des classes (`class_weight='balanced'`)
- **Explicabilité SHAP** :
  - Importance globale des variables (bar plots)
  - Explication locale pour chaque prédiction

Grâce à ce pipeline, nous passons d’une simple détection à une **anticipation concrète des instabilités**, avec une **interprétation directe des causes métier**.

## Prérequis

Pour utiliser ce projet, vous devez d'abord créer un environnement virtuel et installer les packages répertoriés dans le fichier `requirements.txt`.

## Installation

1. Cloner le dépôt sur votre machine :
   ```bash
   git clone https://github.com/votre-utilisateur/innovative-anomaly-detector.git
   cd innovative-anomaly-detector

2. Installer les dépendances avec pip
   ```bash
   pip install -r requirements.txt

## Structure des fichiers
``` 
/Scripts                    # Scripts Python du projet
    ├── EDA.py                         # Analyse exploratoire
    └── Modèles/                       # Détection et modélisation
        ├── Modele_innovant.py             # NeuroGraph STiLe (STL + graphe + SHAP)
        ├── Modele_par_noeud.py            # IsolationForest par nœud (peag, olt, boucle)
        ├── Modele_par_OLT_PEAG.py         # Détection locale par croisement OLT-PEAG
        ├── Modele_supervise.py            # Modèle supervisé + SHAP
        └── Serie_temporelle_OLT_PEAG.py   # Anomalies par moyennes mobiles
/README.md                  # Description du projet
/Requirements.txt           # Dépendances Python 

```
## Contributeurs
- **Salma BENMOUSSA**
- **Charlotte CEGARRA**
- **Chirine DEXPOSITO**
- **Hella BOUHADDA**

Ce projet a été développé dans le cadre du Master MOSEF, à l'université Paris 1 Panthéon Sorbonne, en partenariat avec SFR et le cabinet de consulting Nexialog.

## 📩 Contact

N'hésitez pas à nous contacter pour toute question :

- salmabenmoussa103@gmail.com 
- charlottecegarrapro@gmail.com
- chirinedexposito@gmail.com
- hella.bouhadda@gmail.com
