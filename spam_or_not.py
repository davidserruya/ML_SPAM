import pickle
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
import pandas as pd
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer 
tfidf = TfidfVectorizer(stop_words='english')
stopwords = stopwords.words('english')
stemmer = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()
nltk.download('wordnet')

file = open('models.pkl', 'rb')
table= pickle.load(file)
file.close()
modelSVM=table[0]
modelNB=table[1]
modelKNN=table[2]
tfidf=table[3]
tableau_text=[]

def clean_text(text):
    new_text=text.lower()
    clean_text= re.sub("[^a-z]+"," ",new_text)
    clean_text_stopwords = ""
    for i in clean_text.split(" ")[1:]:
        if not i in stopwords and len(i) > 3:
            wordlem=lemmatizer.lemmatize(i)
            wordStem=stemmer.stem(wordlem)
            clean_text_stopwords += wordStem
            clean_text_stopwords += " "
    return clean_text_stopwords

st.markdown("<h1 style='text-align: center; color: red;'>SPAM OR NOT ?</h1>", unsafe_allow_html=True)

# Affichage barre latérale
st.sidebar.markdown("<h1 style='text-align: center; color: red;'>MAIL MENU</h1>", unsafe_allow_html=True)
mail_text = st.sidebar.text_area("Entrez le contenu texte de votre mail :")
option = st.sidebar.selectbox(
     'Quel algorithme choisissez-vous ?',
     ('SVM', 'Naive Bayes', 'KNN'))
button = st.sidebar.button("Lancer")
# Fin affichage barre latérale

if button:
     st.write(mail_text)
     tableau_text.append(clean_text(mail_text))
     if(option=='SVM'):
           predicted= modelSVM.predict(tfidf.transform(tableau_text))
           resultat=predicted[0]
     elif(option=='Naive Bayes'):
           predicted= modelNB.predict(tfidf.transform(tableau_text))
           resultat=predicted[0]
     elif(option=='KNN'): 
           predicted= modelKNN.predict(tfidf.transform(tableau_text))
           resultat=predicted[0]
     if(resultat==0):
          st.write("not spam")
     elif(resultat==1):
          st.write("Spam")  
     tableau_text=[]
