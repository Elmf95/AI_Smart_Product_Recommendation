import pandas as pd
from scipy.sparse import csr_matrix, save_npz

# Chemin vers le fichier filtré
input_file = r"C:\data\filtered_reviews.csv"

# Chemin pour enregistrer la matrice sparse
output_sparse_file = r"C:\data\user_product_sparse.npz"

try:
    # Lecture du fichier filtré
    print("Lecture du fichier filtré...")
    data = pd.read_csv(input_file, sep=',', encoding='utf-8')

    # Vérification des colonnes nécessaires
    required_columns = {'customer_id', 'product_id', 'star_rating'}
    if not required_columns.issubset(data.columns):
        raise ValueError(f"Le fichier doit contenir les colonnes : {required_columns}")

    # Encodage des IDs en indices uniques
    print("Encodage des IDs en indices uniques...")
    user_ids = data['customer_id'].astype("category").cat.codes
    product_ids = data['product_id'].astype("category").cat.codes

    # Extraction des ratings
    ratings = data['star_rating'].astype(float)

    # Création de la matrice sparse
    print("Construction de la matrice utilisateur-produit sparse...")
    sparse_matrix = csr_matrix((ratings, (user_ids, product_ids)))

    # Sauvegarde de la matrice sparse
    print("Enregistrement de la matrice sparse...")
    save_npz(output_sparse_file, sparse_matrix)
    print(f"Matrice sparse enregistrée avec succès dans : {output_sparse_file}")

    # Informations sur la matrice
    print("\nInformations sur la matrice sparse :")
    print(f"Dimensions : {sparse_matrix.shape}")
    print(f"Nombre d'éléments non nuls : {sparse_matrix.nnz}")

except Exception as e:
    print(f"Une erreur est survenue : {e}")
