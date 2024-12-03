import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Chargement des données
input_file = r"C:\data\filtered_reviews.csv"

try:
    # Lecture du fichier CSV
    data = pd.read_csv(input_file, encoding='utf-8')

    # Conversion de la colonne 'review_date' en type datetime
    data['review_date'] = pd.to_datetime(data['review_date'])
    

    # 1. Analyse des produits les plus évalués
    top_products = data['product_id'].value_counts().head(10)
    print("Top 10 des produits les plus évalués :")
    print(top_products)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_products.index, y=top_products.values, palette='muted')
    plt.title("Top 10 des produits les plus évalués")
    plt.xlabel("ID Produit")
    plt.ylabel("Nombre d'évaluations")
    plt.xticks(rotation=45)
    plt.show()
    
    # 1. Distribution des notes
    plt.figure(figsize=(8, 5))
    sns.countplot(x='star_rating', data=data, palette='viridis')
    plt.title("Distribution des notes")
    plt.xlabel("Étoiles")
    plt.ylabel("Nombre de notes")
    plt.show()


    # 2. Analyse des utilisateurs les plus actifs
    top_users = data['customer_id'].value_counts().head(10)
    print("Top 10 des utilisateurs les plus actifs :")
    print(top_users)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_users.index, y=top_users.values, palette='coolwarm')
    plt.title("Top 10 des utilisateurs les plus actifs")
    plt.xlabel("ID Utilisateur")
    plt.ylabel("Nombre d'évaluations")
    plt.xticks(rotation=45)
    plt.show()

    # 3. Tendances temporelles des évaluations
    data['year_month'] = data['review_date'].dt.to_period('M')
    reviews_per_month = data.groupby('year_month').size()

    # Nombre d'évaluations par mois
    plt.figure(figsize=(12, 6))
    reviews_per_month.plot(kind='line', marker='o', color='blue')
    plt.title("Nombre d'évaluations par mois")
    plt.xlabel("Date (Année-Mois)")
    plt.ylabel("Nombre d'évaluations")
    plt.grid()
    plt.show()

    # Moyenne des notes par mois
    average_rating_per_month = data.groupby('year_month')['star_rating'].mean()

    plt.figure(figsize=(12, 6))
    average_rating_per_month.plot(kind='line', marker='o', color='green')
    plt.title("Évolution de la note moyenne par mois")
    plt.xlabel("Date (Année-Mois)")
    plt.ylabel("Note moyenne")
    plt.grid()
    plt.show()

    # 4. Analyse de la sparsité des interactions
    total_users = data['customer_id'].nunique()
    total_products = data['product_id'].nunique()
    total_interactions = len(data)

    sparsity = 1 - (total_interactions / (total_users * total_products))
    print(f"Sparsité de la matrice utilisateur-produit : {sparsity:.2%}")

except Exception as e:
    print(f"Une erreur est survenue : {e}")
