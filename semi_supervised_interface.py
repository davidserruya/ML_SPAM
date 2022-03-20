import streamlit as st
from PIL import Image

file = open('metrics_semi_supervised.pkl', 'rb')
metrics_semi_supervised= pickle.load(file)
file.close()

st.markdown("<h1 style='text-align: center; color: red;'>SSL INTERFACE</h1>", unsafe_allow_html=True)
st.markdown(
"""
En apprentissage semi-supervisé, un algorithme exploite de grandes quantités de données non étiquetées disponibles dans de nombreux cas d'utilisation en combinaison avec des ensembles de données étiquetées généralement plus petits.
""")
st.markdown(
"""
Pour notre étude, nous avons choisi deux algorithmes différents:
* K-moyennes (Kmeans) : est est une méthode de quantification vectorielle , issue du traitement du signal , qui vise à partitionner n observations en k clusters dans lesquels chaque observation appartient au cluster dont la moyenne est la plus proche.
* Regroupement hiérarchique : est une méthode avce deux approches possibles, ascendante et descendante. 
""")
image = Image.open('semisup.png')
st.image(image)
st.markdown(
"""
Dans notre cas, notre dataset est composé de mails où on ne connait pas leur nature. Nous allons répartir notre dataset en 80% d'entrainement et 20% de test.    
Pour chaque algorithme, nous allons l'entraîner et lui permettre de trouver des similarités afin de définir deux groupes.
Ensuite nous allons tester le modèle avec le set de test et pouvoir calculer la précision, le rappel et le taux de reconnaissance de chaque modèle.  
Ces indicateurs vont nous permettre de déterminer l'algorithme le plus efficace pour ce problème.
""")
st.dataframe(metrics_semi_supervised)
