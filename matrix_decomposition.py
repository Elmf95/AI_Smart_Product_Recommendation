import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate
import streamlit as st

# Chemin vers le fichier filtré
input_file = r"C:\data\filtered_reviews.csv"

# Chargement des données
data = pd.read_csv(input_file, encoding='utf-8')

# Préparer les données pour Surprise
reader = Reader(rating_scale=(1, 5))
surprise_data = Dataset.load_from_df(data[['customer_id', 'product_id', 'star_rating']], reader)

# Initialiser et évaluer le modèle SVD
st.title("Système de Recommandation avec SVD")

@st.cache_resource
def train_svd():
    model = SVD()
    cross_validate(model, surprise_data, cv=5, verbose=False)  # Validation croisée
    trainset = surprise_data.build_full_trainset()
    model.fit(trainset)
    return model

model = train_svd()

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

# Interface utilisateur
st.write("Obtenez des recommandations en fonction de vos évaluations passées.")

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
