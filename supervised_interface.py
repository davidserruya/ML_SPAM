from supervised_algo import df_performance_metrics
from PIL import Image
import streamlit as st



# Affichage page principale
st.markdown("<h1 style='text-align: center; color: red;'>SL INTERFACE</h1>", unsafe_allow_html=True)
st.markdown(
"""
En apprentissage supervisé, un algorithme reçoit un ensemble de données qui est étiquetées avec des valeurs de sorties correspondantes sur lequel il va pouvoir s’entraîner et définir un modèle de prédiction (training set). Cet algorithme pourra par la suite être utilisé sur de nouvelles données afin de prédire leurs valeurs de sorties correspondantes (testing set).

""")
st.markdown(
"""
Pour notre étude, nous avons choisi trois algorithmes différents:
* Machine à vecteurs de support (SVM) : est décrit géométriquement comme un hyperplan qui sépare de façon optimale le document en deux groupes de données, d’un côté l’opinion positive (spam) et de l’autre l’opinion négative (non spam).
* Classificatio naïve bayésienne : est une méthode de classification statistique qui peut être utilisée pour prédire la probabilité d'appartenance à une classe, dans notre cas il existe deux classes : spam ou non-spam.
* K plus proches voisins (KNN) : suppose que des objets similaires existent à proximité dans cet espace (plus proches voisins).  En d'autres termes, des choses similaires sont proches les unes des autres. 

""")
image = Image.open('sup.png')
st.image(image)
st.markdown(
"""
Dans notre cas, notre dataset est composé de mails qui pour certains sont des spams. Nous allons répartir notre dataset en 80% d'entrainement et 20% de test.    
Pour chaque algorithme, nous allons l'entraîner et lui permettre de définir un modèle de prédiction grâce au jeu d'entraînement.
Ensuite nous allons tester le modèle avec le set de test et pouvoir calculer la précision, le rappel et le taux de reconnaissance de chaque modèle.  
Ces indicateurs vont nous permettre de déterminer l'algorithme le plus efficace pour ce problème.
""")
st.write("")
st.dataframe(df_performance_metrics)
