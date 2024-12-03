import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les données filtrées
file_path = r"C:\data\filtered_reviews.csv"
data = pd.read_csv(file_path, parse_dates=['review_date'])

# 1. Distribution des notes
plt.figure(figsize=(8, 5))
sns.countplot(x='star_rating', data=data, palette='viridis')
plt.title("Distribution des notes")
plt.xlabel("Étoiles")
plt.ylabel("Nombre de notes")
plt.show()

# 2. Top produits les plus évalués
top_products = data['product_id'].value_counts().head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_products.index, y=top_products.values, palette='muted')
plt.title("Top 10 produits les plus évalués")
plt.xlabel("ID Produit")
plt.ylabel("Nombre d'évaluations")
plt.xticks(rotation=45)
plt.show()

# 3. Tendances temporelles
data['year_month'] = data['review_date'].dt.to_period('M')
trend = data.groupby('year_month').size()

plt.figure(figsize=(12, 6))
trend.plot()
plt.title("Tendance des avis par mois")
plt.xlabel("Mois")
plt.ylabel("Nombre d'avis")
plt.grid()
plt.show()

# 4. Sparsité de la matrice utilisateur-produit
unique_users = data['customer_id'].nunique()
unique_products = data['product_id'].nunique()
total_interactions = len(data)

sparsity = 1 - (total_interactions / (unique_users * unique_products))
print(f"Sparsité de la matrice utilisateur-produit : {sparsity:.2%}")
