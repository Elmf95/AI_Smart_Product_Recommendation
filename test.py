import streamlit as st
import numpy as np
from scipy.sparse import csr_matrix
from lightfm import LightFM

# Création d'une petite matrice sparse pour le test
rows, cols = 100, 100
data = np.random.rand(100)
row_indices = np.random.randint(0, rows, size=100)
col_indices = np.random.randint(0, cols, size=100)
sparse_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(rows, cols))

st.write("Matrice générée")

# Entraînement LightFM
try:
    model = LightFM(loss='warp')
    model.fit(sparse_matrix, epochs=5, num_threads=1)
    st.write("Modèle entraîné avec succès")
except Exception as e:
    st.error(f"Erreur lors de l'entraînement : {e}")
