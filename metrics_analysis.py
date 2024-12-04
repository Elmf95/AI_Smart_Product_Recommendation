import numpy as np
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, auc_score, reciprocal_rank
from scipy.sparse import load_npz, csr_matrix
import logging

# Configuration du journal
logging.basicConfig(level=logging.INFO)

# Chemin pour charger la matrice sparse
sparse_file = r"C:\data\user_product_sparse.npz"

# Charger la matrice utilisateur-produit sparse
try:
    data_matrix = load_npz(sparse_file)
    logging.info(f"Matrice chargée avec succès, dimensions : {data_matrix.shape}")

    # Limitation à une matrice de 50,000 x 50,000
    max_users, max_items = 50000, 50000
    data_matrix = data_matrix[:max_users, :max_items]
    logging.info(f"Matrice tronquée à : {data_matrix.shape}")
except Exception as e:
    raise RuntimeError(f"Erreur lors du chargement de la matrice sparse : {e}")

# Vérifier et corriger les valeurs NaNs et infinis
if np.isnan(data_matrix.data).any() or np.isinf(data_matrix.data).any():
    logging.info("Correction des valeurs invalides dans la matrice...")
    data_matrix.data = np.nan_to_num(data_matrix.data)

# Séparation des données en ensembles d'entraînement et de test
def train_test_split(data_matrix, test_fraction=0.2):
    test_data = data_matrix.copy()
    train_data = data_matrix.copy()

    for user_id in range(data_matrix.shape[0]):
        row_indices = data_matrix[user_id].indices
        test_size = int(len(row_indices) * test_fraction)
        
        if test_size > 0:
            test_indices = np.random.choice(row_indices, size=test_size, replace=False)
            train_data[user_id, test_indices] = 0
            test_data[user_id, np.setdiff1d(row_indices, test_indices)] = 0

    train_data.eliminate_zeros()
    test_data.eliminate_zeros()
    return train_data, test_data

try:
    train_data, test_data = train_test_split(data_matrix)
    logging.info(f"Entraînement : {train_data.shape}, Non-zéros : {train_data.nnz}")
    logging.info(f"Test : {test_data.shape}, Non-zéros : {test_data.nnz}")
except Exception as e:
    raise RuntimeError(f"Erreur lors de la séparation des données : {e}")

# Vérifier les ensembles pour des valeurs invalides
if np.isnan(train_data.data).any() or np.isinf(train_data.data).any():
    raise ValueError("L'ensemble d'entraînement contient des valeurs invalides.")
if np.isnan(test_data.data).any() or np.isinf(test_data.data).any():
    raise ValueError("L'ensemble de test contient des valeurs invalides.")

# Assurer que les matrices sont en format CSR (Compressed Sparse Row)
train_data = csr_matrix(train_data)
test_data = csr_matrix(test_data)

# Initialiser et entraîner le modèle
model = LightFM(loss='warp')
logging.info("Entraînement du modèle LightFM...")
try:
    model.fit(train_data, epochs=5, num_threads=2)
    logging.info("Modèle entraîné avec succès.")
except Exception as e:
    raise RuntimeError(f"Erreur lors de l'entraînement du modèle : {e}")

# Vérification rapide des prédictions
logging.info("Test de prédictions...")

# Prédire quelques résultats pour vérifier que le modèle fonctionne
sample_user_id = 0  # Choisir un utilisateur de test
scores = model.predict(sample_user_id, np.arange(test_data.shape[1]))
logging.info(f"Prédictions pour l'utilisateur {sample_user_id} : {scores[:10]}")  # Afficher les 10 premières prédictions

# Calcul des métriques
def evaluate_model(model, train_data, test_data, k=10):
    logging.info("Début de l'évaluation du modèle...")

    try:
        # Calcul des métriques Precision@K, AUC et MRR
        precision = precision_at_k(model, test_data, train_interactions=train_data, k=k).mean()
        logging.info(f"Precision@K : {precision:.4f}")

        auc = auc_score(model, test_data, train_interactions=train_data).mean()
        logging.info(f"AUC : {auc:.4f}")

        mrr = reciprocal_rank(model, test_data, train_interactions=train_data).mean()
        logging.info(f"MRR : {mrr:.4f}")

    except Exception as e:
        logging.error(f"Erreur lors du calcul des métriques : {e}")
        raise RuntimeError(f"Erreur lors du calcul des métriques : {e}")
    
    return {"Precision@K": precision, "AUC": auc, "MRR": mrr}

# Évaluation
try:
    metrics = evaluate_model(model, train_data, test_data, k=10)
    logging.info("\nRésultats d'évaluation :")
    for metric, value in metrics.items():
        logging.info(f"{metric} : {value:.4f}")
except Exception as e:
    raise RuntimeError(f"Erreur lors de l'évaluation du modèle : {e}")
