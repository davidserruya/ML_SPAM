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


dataset = pd.read_csv("spam_clean.csv")
x = dataset['text_clean'].values.astype('U')
y = dataset['label_num']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2)

models = {
    'Kmeans1': MiniBatchKMeans(),
    'Kmeans2':KMeans(),
    'AC':AgglomerativeClustering(),
}

params = {
    'Kmeans1': { 'n_clusters':[2], 'random_state':[0]},
    'Kmeans2': { 'n_clusters': [2], 'init' :['k-means++']},
    'AC': { 'n_clusters':[2]},    
}


def ML_modeling(models, params, X_train, X_test, y_test):    
    
    if not set(models.keys()).issubset(set(params.keys())):
        raise ValueError('Some estimators are missing parameters')
    
    performance_metrics=[]
    for key in models.keys():
        model = models[key]
        param = params[key]
        if(str(model)=='AgglomerativeClustering()'):
            X_train=X_train.toarray()
            X_test=X_test.toarray()
            model = AgglomerativeClustering(n_clusters=2).fit(X_train)
            y_prediction=model.fit_predict(X_test)
        else:
            gs = GridSearchCV(model, param, cv=10, error_score=0, refit=True) 
            gs.fit(X_train)
            y_prediction = gs.predict(X_test)
            
        
        # Print scores for the classifier
        accuracy_sc = accuracy_score(y_test, y_prediction)
        precision_sc= precision_score(y_test, y_prediction, average='macro')
        recall_sc = recall_score(y_test, y_prediction, average='macro')
        f1_sc =  f1_score(y_test, y_prediction, average='macro')
        #print(confusion_matrix(y_test, y_prediction))
        
        
        performance_metrics.append([key,accuracy_sc,precision_sc,recall_sc,f1_sc])
        #print(key, ':', gs.best_params_)
        #print("Accuracy: %1.3f \tPrecision: %1.3f \tRecall: %1.3f \t\tF1: %1.3f\n" % (accuracy_sc, precision_sc, recall_sc, f1_sc))
    return pd.DataFrame(performance_metrics,columns=['Model' , 'Accuracy', 'Precision' , 'Recall', "F1 Score"])


tfidf = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf.fit_transform(x_train)
X_test_tfidf = tfidf.transform(x_test)
df_performance_metrics=ML_modeling(models, params, X_train_tfidf, X_test_tfidf, y_test)