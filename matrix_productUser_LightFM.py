import gradio as gr
import numpy as np
from scipy.sparse import load_npz, csr_matrix
from lightfm import LightFM
import os
import pandas as pd  # Ajout de pandas

# Chemin pour charger la matrice sparse
sparse_file = r"C:\data\user_product_sparse.npz"

# Chemin pour enregistrer le feedback utilisateur
feedback_folder = r"C:\data\csv_output"
feedback_file = os.path.join(feedback_folder, "user_feedback.csv")

# Vérifier si le dossier existe, sinon le créer
if not os.path.exists(feedback_folder):
    os.makedirs(feedback_folder)

# Fonction pour réduire la matrice si elle est trop grande
def reduce_sparse_matrix(matrix, max_users=50000, max_items=50000):
    return matrix[:max_users, :max_items]

# Chargement de la matrice utilisateur-produit
def load_matrix():
    try:
        sparse_matrix = load_npz(sparse_file)

        # Dimensions originales de la matrice
        original_shape = sparse_matrix.shape
        print(f"Dimensions originales de la matrice : {original_shape}")

        # Vérification des valeurs NaN/infinies
        if not np.isfinite(sparse_matrix.data).all():
            print("La matrice contient des valeurs non finies. Nettoyage en cours...")
            sparse_matrix.data = np.nan_to_num(sparse_matrix.data)

        # Réduction pour éviter les problèmes de mémoire
        if sparse_matrix.shape[0] > 50000 or sparse_matrix.shape[1] > 50000:
            print("Réduction de la matrice pour l'entraînement...")
            sparse_matrix = reduce_sparse_matrix(sparse_matrix)

        return sparse_matrix

    except Exception as e:
        print(f"Erreur lors du chargement de la matrice : {e}")
        return None

# Chargement de la matrice utilisateur-produit
sparse_matrix = load_matrix()

# Configuration et entraînement du modèle LightFM
model = LightFM(loss='warp')
for epoch in range(5):  # Limitez le nombre d'époques pour tester
    model.fit_partial(sparse_matrix, epochs=1, num_threads=4)
    print(f"Époque {epoch + 1} terminée.")

# Fonction pour recommander des produits
def recommend_products(user_id, n_recommendations=10):
    n_users, n_items = sparse_matrix.shape
    scores = model.predict(user_id, np.arange(n_items))
    top_items = np.argsort(-scores)[:n_recommendations]
    return top_items

# Fonction pour enregistrer le feedback
def save_feedback(user_id, feedback):
    feedback_data = {
        "user_id": [user_id],
        "feedback": [feedback],
        "loss_mode": ["warp"],  # Méthode utilisée
    }

    if os.path.exists(feedback_file):
        feedback_df = pd.read_csv(feedback_file)
        feedback_df = pd.concat([feedback_df, pd.DataFrame(feedback_data)], ignore_index=True)
    else:
        feedback_df = pd.DataFrame(feedback_data)

    feedback_df.to_csv(feedback_file, index=False)

# Fonction principale de l'interface Gradio
def recommend_and_save_feedback(user_id, feedback):
    try:
        recommended_items = recommend_products(user_id)
        recommendations = [f"Produit {item}" for item in recommended_items]
        save_feedback(user_id, feedback)
        return recommendations, f"Feedback enregistré avec succès : {feedback}"
    except Exception as e:
        return [], f"Erreur lors de la génération des recommandations : {e}"

# HTML pour le titre personnalisé
custom_title = """
<h1><b>Système de recommandations de produits</b></h1>
<p style="font-size:14px; color:gray;">Matrice réduite à 50 000 utilisateurs et 50 000 produits par rapport à la taille initiale de 796 587 utilisateurs et 699 450 produits afin d'économiser de la mémoire (ne pas dépasser 49 999).</p>
"""

# Interface Gradio
interface = gr.Interface(
    fn=recommend_and_save_feedback,
    inputs=[
        gr.Number(label="ID Utilisateur", value=0, precision=0, minimum=0, maximum=sparse_matrix.shape[0] - 1),
        gr.Radio(["Oui", "Non"], label="Les recommandations sont-elles pertinentes ?")
    ],
    outputs=[
        gr.Textbox(label="Recommandations", interactive=False),
        gr.Textbox(label="Feedback", interactive=False)
    ],
    live=True,
    title=custom_title,  # Utilisation du titre personnalisé
)

# Lancement de l'interface Gradio
interface.launch(share=True)
