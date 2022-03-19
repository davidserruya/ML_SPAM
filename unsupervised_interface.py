import pickle
from PIL import Image
import streamlit as st

file = open('metrics_unsupervised.pkl', 'rb')
metrics_unsupervised= pickle.load(file)
file.close()

# Affichage page principale
st.markdown("<h1 style='text-align: center; color: red;'>UL INTERFACE</h1>", unsafe_allow_html=True)
st.markdown(
"""
En apprentissage non-supervisé, un algorithme reçoit un ensemble de données non-étiquetées sur lequel il va pouvoir s’entraîner et identifier des similarités afin de réaliser des regroupements.

""")
st.markdown(
"""
Pour notre étude, nous avons choisi deux algorithmes différents:
* K-moyennes (Kmeans) : est est une méthode de quantification vectorielle , issue du traitement du signal , qui vise à partitionner n observations en k clusters dans lesquels chaque observation appartient au cluster dont la moyenne est la plus proche.
* Regroupement hiérarchique : est une méthode avce deux approches possibles, ascendante et descendante. 
""")
image = Image.open('unsup.png')
st.image(image)
st.markdown(
"""
Dans notre cas, notre dataset est composé de mails où on ne connait pas leur nature. Nous allons répartir notre dataset en 80% d'entrainement et 20% de test.    
Pour chaque algorithme, nous allons l'entraîner et lui permettre de trouver des similarités afin de définir deux groupes.
Ensuite nous allons tester le modèle avec le set de test et pouvoir calculer la précision, le rappel et le taux de reconnaissance de chaque modèle.  
Ces indicateurs vont nous permettre de déterminer l'algorithme le plus efficace pour ce problème.
""")
st.write("")
st.dataframe(metrics_unsupervised)
