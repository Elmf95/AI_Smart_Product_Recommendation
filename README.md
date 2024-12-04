# Système de Recommandations avec LightFM

Ce projet implémente un système de recommandations contextuelles pour des produits, basé sur les évaluations de clients d'Amazon. Le pipeline couvre l'extraction des données brutes, leur transformation en une matrice utilisateur-produit sparse, l'entraînement d'un modèle LightFM et l'intégration d'une interface utilisateur interactive via Gradio.

## Fonctionnalités principales

1. **Lecture et prétraitement des données** :
   - Chargement des évaluations Amazon depuis un fichier TSV.
   - Filtrage des colonnes nécessaires pour réduire la taille des données.
   - Sauvegarde des données filtrées au format CSV.

2. **Analyse exploratoire des données** :
   - Analyse des produits les plus évalués.
   - Distribution des notes.
   - Identification des utilisateurs les plus actifs.
   - Analyse temporelle et calcul de la sparsité de la matrice utilisateur-produit.

3. **Construction de la matrice utilisateur-produit sparse** :
   - Conversion des identifiants utilisateurs et produits en indices uniques.
   - Création d'une matrice sparse avec les évaluations.
   - Sauvegarde de la matrice sparse au format `.npz`.

4. **Entraînement du modèle LightFM** :
   - Utilisation de l'algorithme WARP (Weighted Approximate-Rank Pairwise).
   - Évaluation du modèle avec les métriques de Précision@K et Rappel@K.

5. **Interface interactive avec Gradio** :
   - Recommandation de produits pour un utilisateur donné.
   - Collecte du feedback utilisateur et sauvegarde.

## Organisation des scripts

- **read_and_filter.py** : Prépare les données initiales en les filtrant et les sauvegarde en CSV.
- **data_analysis.py** : Fournit une analyse exploratoire des données filtrées.
- **matrix_construct.py** : Construit une matrice utilisateur-produit sparse à partir des données filtrées.
- **matrix_user_productLightFM.py** : Entraîne le modèle LightFM et lance une interface utilisateur interactive.

## Prérequis

### Outils et technologies
- **Python 3.8 ou supérieur**
- Bibliothèques Python : `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `lightfm`, `gradio`

### Installation des dépendances
Assurez-vous d'avoir installé toutes les dépendances avant d'exécuter les scripts. Vous pouvez utiliser la commande suivante :
```bash
pip install pandas numpy matplotlib seaborn scipy lightfm gradio
```

## Instructions d'exécution

(Vous n'aurez besoin d'utiliser que la partie 4 pour faire fonctionner le projet);

### 1. Prétraitement des données
Exécutez le script `read_and_filter.py` pour charger, filtrer et sauvegarder les données prétraitées :
```bash
python read_and_filter.py
```

### 2. Analyse exploratoire des données
Lancez le script `data_analysis.py` pour analyser les données filtrées et visualiser les résultats :
```bash
python data_analysis.py
```

### 3. Construction de la matrice utilisateur-produit sparse
Utilisez le script `matrix_construct.py` pour créer et sauvegarder la matrice sparse :
```bash
python matrix_construct.py
```

### 4. Entraînement et interface interactive
Lancez `matrix_user_productLightFM.py` pour entraîner le modèle LightFM et utiliser l'interface utilisateur interactive :
```bash
python matrix_user_productLightFM.py
```
L'interface Gradio s'ouvrira dans un navigateur ou proposera un lien partageable.


## Structure du dépôt

```plaintext
/
├── read_and_filter.py
├── data_analysis.py
├── matrix_construct.py
├── matrix_user_productLightFM.py
├── requirements.txt
├── data/
│   ├── amazon_reviews_us_Digital_Music_Purchase_v1_00.tsv
│   ├── filtered_reviews.csv
│   ├── user_product_sparse.npz
```

## Notes
- Vérifiez que le fichier TSV initial est bien placé dans le répertoire du repository.
- Réduisez la matrice si vous rencontrez des problèmes de mémoire.
- Modifiez les hyperparamètres du modèle LightFM pour ajuster les performances.

## Auteur
Projet développé par Fikri El mehdi dans le cadre d'une implémentation pratique de systèmes de recommandations.
