import os
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate
import streamlit as st
import joblib

# Déterminer le chemin du fichier dynamique
root_dir = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(root_dir, "filtered_reviews.csv")

# Vérifier si le fichier existe
if not os.path.exists(input_file):
    raise FileNotFoundError(f"Le fichier 'filtered_reviews.csv' est introuvable dans le répertoire racine : {root_dir}")

# Chargement des données
data = pd.read_csv(input_file, encoding='utf-8')

# Préparer les données pour Surprise
reader = Reader(rating_scale=(1, 5))
surprise_data = Dataset.load_from_df(data[['customer_id', 'product_id', 'star_rating']], reader)

# Fonction de sauvegarde du modèle
def save_model(model):
    joblib.dump(model, 'svd_model.pkl')

# Fonction de chargement du modèle
@st.cache_resource
def load_svd_model():
    model_path = 'svd_model.pkl'
    
    # Vérifier si le modèle est déjà sauvegardé
    if os.path.exists(model_path):
        # Charger le modèle déjà entraîné
        model = joblib.load(model_path)
        return model
    else:
        # Si le modèle n'existe pas, entraîner un nouveau modèle
        model = train_svd()
        save_model(model)
        return model

# Fonction d'entraînement du modèle SVD
def train_svd():
    model = SVD()
    # Validation croisée sur un sous-ensemble des données pour accélérer
    small_data = Dataset.load_from_df(data[['customer_id', 'product_id', 'star_rating']].sample(n=5000), reader)
    cross_validate(model, small_data, cv=3, verbose=False)  # Validation croisée réduite
    trainset = surprise_data.build_full_trainset()
    model.fit(trainset)
    return model

# Charger le modèle (ou l'entraîner si nécessaire)
model = load_svd_model()

# Fonction de recommandations
def recommend_products(user_id, top_n=10):
    user_rated_products = set(data[data['customer_id'] == user_id]['product_id'])
    all_products = data['product_id'].unique()
    predictions = [
        (product, model.predict(user_id, product).est)
        for product in all_products if product not in user_rated_products
    ]
    top_recommendations = sorted(predictions, key=lambda x: x[1], reverse=True)[:top_n]
    return top_recommendations

# Interface utilisateur Streamlit
st.title("Système de Recommandation avec SVD")

st.write("Obtenez des recommandations en fonction de vos évaluations passées.")

# Demander l'ID utilisateur
user_id_input = st.text_input("Entrez votre ID utilisateur :")

if user_id_input:
    try:
        user_id_input = int(user_id_input)
        if user_id_input not in data['customer_id'].values:
            st.error("L'utilisateur n'existe pas dans la base de données.")
        else:
            recommendations = recommend_products(user_id_input)
            st.subheader(f"Recommandations pour l'utilisateur {user_id_input} :")
            for product, rating in recommendations:
                st.write(f"- **Produit** : {product}, **Note prédite** : {rating:.2f}")
    except Exception as e:
        st.error(f"Erreur lors de la génération des recommandations : {e}")
