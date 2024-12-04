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

  
### Installer un compilateur C

**LightFM** nécessite un compilateur C pour construire certaines de ses parties. Sur Windows, vous devez installer **Build Tools for Visual Studio**. Suivez ces étapes :

 **Téléchargez et installez les outils de construction de Visual Studio** depuis ce [lien officiel](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

   Pendant l’installation, sélectionnez :
   - **Desktop development with C++**.

   Assurez-vous que les options suivantes sont cochées :
   - **MSVC v142**.
   - **Windows 10 SDK**.

   **Redémarrez votre terminal** après l'installation.



### Installation des dépendances
Assurez-vous d'avoir installé toutes les dépendances avant d'exécuter les scripts. Vous pouvez utiliser la commande suivante :
```bash
pip install pandas numpy matplotlib seaborn scipy lightfm gradio
ou directement grâce à :
pip install -r requirements.txt
```

## Instructions d'exécution

(Vous n'aurez besoin d'utiliser que la partie 4 et 5 pour faire fonctionner le projet);

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

### 4. Script `matrix_deconstruct.py`

## Rôle
Ce script implémente un **système de recommandation de produits** basé sur la **décomposition matricielle (SVD)**. Il utilise les évaluations passées des utilisateurs pour prédire leurs préférences et afficher des recommandations via une interface **Streamlit**.

## Utilisation

1. Téléchargez le fichier de données `filtered_reviews.csv` via ce lien MEGA :  
   [Télécharger les données](https://mega.nz/file/CUkjVIBL#yDZ7bl78onP2LrV8qE2idg01mES7klB22XeA9g8kKpg) 
   Placez-le dans le dossier `racine` du projet.

   Pour lancer le script utiliser cette commande.
   streamlit run matrix_deconstruct.py

   Puis ouvrez le fichier csv sous format excel et choisissez un ID d'utilisateur proposé par la table.




### 5. Entraînement et interface interactive
Lancez `matrix_user_productLightFM.py` pour entraîner le modèle LightFM et utiliser l'interface utilisateur interactive :

Ce script implémente un système de recommandation de produits utilisant l'algorithme LightFM. Il fait des recommandations personnalisées pour chaque utilisateur à partir d'une matrice sparse des interactions utilisateur-produit, en utilisant la méthode WARP

Pour lancer le script utilisez cette commande après avoir vérifié que le fichier est bien dans la racine du projet, si le git clone s'est bien passé il y sera.
```bash
python matrix_user_productLightFM.py
```
L'interface Gradio vrira dans un navigateur ou proposera un lien partageableoù vous pourrez :

Entrer l'ID d'un utilisateur pour générer des recommandations de produits.
Donner votre feedback sur la pertinence des recommandations ("Oui" ou "Non").
Le feedback utilisateur est ensuite enregistré dans un fichier CSV (user_feedback.csv) pour un suivi des préférences


## Structure du dépôt

```plaintext
/
├── read_and_filter.py
├── data_analysis.py
├── matrix_construct.py
├── matrix_user_productLightFM.py
├── requirements.txt
│   ├── filtered_reviews.csv
│   ├── user_product_sparse.npz
```

## Notes
- Vérifiez que le fichier TSV initial est bien placé dans le répertoire du repository.
- Vérifiez que le fichier CSV initial est bien placé dans le répertoire du repository après téléchargement.
- Réduisez la matrice si vous rencontrez des problèmes de mémoire.
- Modifiez les hyperparamètres du modèle LightFM pour ajuster les performances.

## Auteur
Projet développé par Fikri El mehdi dans le cadre d'une implémentation pratique de systèmes de recommandations.
