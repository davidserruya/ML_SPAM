import streamlit as st
import pickle
file = open('models.pkl', 'rb')
table= pickle.load(file)
file.close()
modelSVM=table[0]
modelNB=table[1]
modelKNN=table[2]
modelKM1=table[3]
modelKM2=table[4]
modelAC=table[5]

st.markdown("<h1 style='text-align: center; color: red;'>SPAM OR NOT ?</h1>", unsafe_allow_html=True)

# Affichage barre latérale
st.sidebar.markdown("<h1 style='text-align: center; color: red;'>MAIL MENU</h1>", unsafe_allow_html=True)
mail_text = st.sidebar.text_area("Entrez le contenu texte de votre mail :")
option = st.sidebar.selectbox(
     'Quel algorithme choisissez-vous ?',
     ('SVM', 'Naive Bayes', 'KNN','MiniBatchKmeans','Kmeans','AC'))
# Fin affichage barre latérale

if mail_text is not None:
     if(option=='SVM'):
           predicted= modelSVM.predict(mail_text)
           resultat=predicted[0]
     elif(option=='Naive Bayes'):
           predicted= modelNB.predict(mail_text)
           resultat=predicted[0]
     elif(option=='KNN'): 
           predicted= modelKNN.predict(mail_text)
           resultat=predicted[0]
     elif(option=='MiniBatchKmeans'):
           predicted= modelKM1.predict(mail_text)
           resultat=predicted[0]
     elif(option=='Kmeans'):
           predicted= modelKM2.predict(mail_text)
           resultat=predicted[0]
     elif(option=='AC'):
           predicted= modelAC.predict(mail_text)
           resultat=predicted[0]
     if(resultat==0):
          st.write("not spam)
     elif(resultat==1):
          st.write("not spam)           
