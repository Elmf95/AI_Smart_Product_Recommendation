import pandas as pd

# Chemin vers le fichier téléchargé
input_file = r"C:\data\amazon_reviews_us_Digital_Music_Purchase_v1_00.tsv"

try:
    # Chargement du fichier TSV en ignorant les lignes mal formées
    data = pd.read_csv(input_file, sep='\t', encoding='utf-8', on_bad_lines='skip')

    # Affichage des premières lignes pour comprendre la structure des données
    print("Aperçu des données :")
    print(data.head())

    # Informations sur le dataset (colonnes, types de données, valeurs manquantes)
    print("\nInformations sur le dataset :")
    print(data.info())

    # Filtrage des colonnes nécessaires
    filtered_data = data[['customer_id', 'product_id', 'star_rating', 'review_body', 'review_date']]

    # Affichage des premières lignes des données filtrées
    print("\nAperçu des données filtrées :")
    print(filtered_data.head())

    # Sauvegarde des données filtrées dans un nouveau fichier CSV
    output_file = r"C:\data\filtered_reviews.csv"
    filtered_data.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\nLes données filtrées ont été sauvegardées dans : {output_file}")

except Exception as e:
    print(f"Une erreur est survenue lors de la lecture ou du traitement du fichier : {e}")
