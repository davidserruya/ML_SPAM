import pandas as pd 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,precision_score, recall_score, f1_score, accuracy_score
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


dataset = pd.read_csv("spam_clean.csv")
x = dataset['text_clean'].values.astype('U')
y = dataset['label_num']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2)

models = {
    'SVM': SVC(),
    'Naive Bayes': MultinomialNB(),
    'KNN': KNeighborsClassifier()
}

params = {
    'SVM': { 'kernel': ['linear', 'rbf'] },
    'Naive Bayes': { 'alpha': [0.5, 1], 'fit_prior': [True, False] },
    'KNN': { 'n_neighbors': [2,3,4,5,6,7,8,9,10] }
}


def ML_modeling(models, params, X_train, X_test, y_train, y_test):    
    
    if not set(models.keys()).issubset(set(params.keys())):
        raise ValueError('Some estimators are missing parameters')
    
    performance_metrics=[]
    for key in models.keys():
    
        model = models[key]
        param = params[key]
        gs = GridSearchCV(model, param, cv=10, error_score=0, refit=True)
        gs.fit(X_train, y_train)
        y_prediction = gs.predict(X_test)
        
        # Print scores for the classifier
        accuracy_sc = accuracy_score(y_test, y_prediction)
        precision_sc= precision_score(y_test, y_prediction, average='macro')
        recall_sc = recall_score(y_test, y_prediction, average='macro')
        f1_sc =  f1_score(y_test, y_prediction, average='macro')
        
        performance_metrics.append([key,accuracy_sc,precision_sc,recall_sc,f1_sc])
        #print(key, ':', gs.best_params_)
        #print("Accuracy: %1.3f \tPrecision: %1.3f \tRecall: %1.3f \t\tF1: %1.3f\n" % (accuracy_sc, precision_sc, recall_sc, f1_sc))
    return pd.DataFrame(performance_metrics,columns=['Model' , 'Accuracy', 'Precision' , 'Recall', "F1 Score"])


tfidf = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf.fit_transform(x_train)
X_test_tfidf = tfidf.transform(x_test)
df_performance_metrics=ML_modeling(models, params, X_train_tfidf, X_test_tfidf, y_train, y_test)
