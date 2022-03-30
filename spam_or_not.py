from sklearn.datasets import fetch_openml
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle
import re
import streamlit as st
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,precision_score, recall_score, f1_score, accuracy_score
from sklearn.semi_supervised import LabelSpreading
from sklearn.semi_supervised import LabelPropagation
from sklearn.model_selection import GridSearchCV
import pickle
import pandas as pd 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,precision_score, recall_score, f1_score, accuracy_score
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,precision_score, recall_score, f1_score, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
import pickle
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
# Fin affichage barre latérale

if mail_text is not None:
     st.write("caca")
     st.write(mail_text)
     tableau_text.append(clean_text(mail_text))
     st.write(tableau_text[0])
     if(option=='SVM'):
           predicted= modelSVM.predict(tfidf.transform(tableau_text))
           resultat=predicted[0]
           st.write(1)
     elif(option=='Naive Bayes'):
           predicted= modelNB.predict(tfidf.transform(tableau_text))
           resultat=predicted[0]
           st.write(2)
     elif(option=='KNN'): 
           predicted= modelKNN.predict(tfidf.transform(tableau_text))
           resultat=predicted[0]
           st.write(3)
     if(resultat==0):
          st.write("not spam")
     elif(resultat==1):
          st.write("Spam")  
     tableau_text=[]
