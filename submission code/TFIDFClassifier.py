from statistics import LinearRegression
from matplotlib.pyplot import axis, cla
import pandas as pd
import numpy as np
from pkg_resources import WorkingSet
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier, LogisticRegression, Ridge
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier, OutputCodeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import RandomizedSearchCV
from imblearn.over_sampling import SMOTE
import nltk
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('omw-1.4')
from nltk.corpus import stopwords
import re
import pickle
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.utils.fixes import loguniform
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import time

#stopword removal and lemmatization
stopwords = stopwords.words('english')
lemmatizer = WordNetLemmatizer()


"""
VISUALISE CLASSIFIER BY
CONFUSION MATRIX
CLASSIFICATION REPORT
SCATTERPLOT
"""
def TFIDFVisualise():
    labels_color_map = {
    0: '#20b2aa', 1: '#ff7373', 2: '#ffe4e1'
    }

    test_csv = pd.read_csv('test_data_Processed.csv') #review rating
    train_csv = pd.read_csv('train_data_Processed.csv') #sentiment
    train_csv = train_csv.sort_values('sentiment')
    train_csv = train_csv.drop(train_csv[train_csv.sentiment == 'positive'].index[int(train_csv.count()['sentiment']/5.5):])
    X = train_csv['review_content'] #column of reviews
    Y = train_csv['sentiment'] #Coresponding sentiment 
    X_test = test_csv["review_content"]
    Y_test = test_csv["Annotator_1"]
    X_train = X.values.astype('U')
    Y_train = Y
    X_test = X_test.values.astype('U')
    Y_test = Y_test.values.astype('U')
    tf_idf = TfidfVectorizer(sublinear_tf=True)
    #applying tf idf to training data
    X_train_tf = tf_idf.fit_transform(X_train) #creating all in tf-idf form
    X_train_tf = tf_idf.transform(X_train)
    smt = SMOTE(random_state=777, k_neighbors=4)
    X_train_tf, Y_train = smt.fit_resample(X_train_tf, Y_train)
    #print(train_csv.sentiment.value_counts())
    #applying tf idf to test data
    X_test_tf = tf_idf.transform(X_test)
    # create k-means model with custom config
    clustering_model = KMeans(
        n_clusters=3,
        max_iter=300,
    )
    
    #filename = 'TFIDFClassifier.sav'
    classifier = SVC(gamma='scale', class_weight='balanced')
    #classifier = BaggingClassifier(base_estimator=classifier, oob_score=True, random_state=101) #Bagging Classifier
    # classifier = RandomForestClassifier()
    classifier.fit(X_train_tf, Y_train)
    labels = []
    Y_pred = classifier.predict(X_test_tf) #['negative','positive'.....,'negative','neutral']
    print(metrics.classification_report(Y_test, Y_pred, target_names=['negative', 'neutral', 'positive']))

    print("Confusion Metrix: ")
    print(metrics.confusion_matrix(Y_test,Y_pred))
    #change Y_pred to 0,1,2
    for label in Y_pred:
        if label == 'negative':
            labels.append(0)
        elif label == 'neutral':
            labels.append(1)
        else:
            labels.append(2)
    #labels = clustering_model.fit_predict(X_train_tf)
    # print labels

    X = X_test_tf.todense() 
    # ----------------------------------------------------------------------------------------------------------------------

    reduced_data = PCA(n_components=2).fit_transform(X)
    # print reduced_data

    fig, ax = plt.subplots()
    for index, instance in enumerate(reduced_data):
        # print instance, index, labels[index]
        pca_comp_1, pca_comp_2 = reduced_data[index]
        color = labels_color_map[labels[index]]
        ax.scatter(pca_comp_1, pca_comp_2, c=color)
    plt.show()



"""
HYPER PARAMETER TUNING
TO OPTIMIZE CLASSIFIER WITH TUNED HYPER PARAMETER.
"""
def hyperParamsTuning():
    # define the space of hyperparameters to search

    # load dataset
    train_csv = pd.read_csv('train_data_Processed.csv') #sentiment
    train_csv = train_csv.sort_values('sentiment')
    train_csv = train_csv.drop(train_csv[train_csv.sentiment == 'positive'].index[int(train_csv.count()['sentiment']/5.5):])
    # split into input and output elements
    X = train_csv['review_content'] #column of reviews
    Y = train_csv['sentiment'] #Coresponding sentiment 
    X_train = X.values.astype('U')
    tf_idf = TfidfVectorizer(sublinear_tf=True)
    #applying tf idf to training data
    X = tf_idf.fit_transform(X_train) #creating all in tf-idf form
    X = tf_idf.transform(X_train)
    print(X.shape, Y.shape)
    # perform optimization
    svc = SVC(class_weight='balanced', cache_size=300)
    distributions = {'C': loguniform(1e0, 1e5),
                     'gamma': loguniform(1e-6, 1e-2),
                     'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
                     'tol':loguniform(1e-6, 1e-2)}
    result = RandomizedSearchCV(svc, distributions, cv=5, n_jobs=10).fit(X,Y)
    # summarizing finding:
    print(result.best_estimator_)
    print(result.best_params_)


"""
FINDING A RIGHT CLASSIFIER
BINARY CLASSIFICATION OF NEGATIVE AND NEUTRAL DATASET
FOR MULTICLASS CLASSIFICATION
"""
def classifierByNN():
    #TESTING
    #read Data
    test_csv = pd.read_csv('test_data_Processed.csv') #review rating
    test_csv_NN = test_csv.drop(test_csv[test_csv.Annotator_1 == 'positive'].index)
    train_csv = pd.read_csv('train_data_Processed.csv') #sentiment
    train_csv_NNegative = train_csv.drop(train_csv[train_csv.sentiment == 'positive'].index)
    X_NN = train_csv_NNegative['review_content'] #column of reviews
    Y_NN = train_csv_NNegative['sentiment'] #Coresponding sentiment 
    X_test = test_csv_NN["review_content"]
    Y_test = test_csv_NN["Annotator_1"]
    X_train_NN = X_NN.values.astype('U')
    Y_train_NN = Y_NN
    X_test = X_test.values.astype('U')
    Y_test = Y_test.values.astype('U')
    tf_idf = TfidfVectorizer(sublinear_tf=True)
    #applying tf idf to training data
    X_train_tf_NN = tf_idf.fit_transform(X_train_NN) #creating all in tf-idf form
    X_train_tf_NN = tf_idf.transform(X_train_NN)
    smt = SMOTE(random_state=777, k_neighbors=4)
    X_train_tf_NN, Y_train_NN = smt.fit_resample(X_train_tf_NN, Y_train_NN)
    #applying tf idf to test data
    X_test_tf = tf_idf.transform(X_test)
    print("(Test) n_samples: %d, n_features: %d" %X_test_tf.shape)
    print("(NN) n_samples: %d, n_features: %d" %X_train_tf_NN.shape)
    estimators = []

    while True:
        print('Input your classifier')
        classifierchoice = input()
        if classifierchoice == 'naive':
            classifier = MultinomialNB()
        elif classifierchoice == 'SGD2':
            classifier = SGDClassifier(loss='log',shuffle=True, warm_start=True)
        elif classifierchoice == 'SGD3':
            classifier = SGDClassifier(loss='modified_huber',shuffle=True)
        elif classifierchoice == 'SGD4':
            classifier = SGDClassifier(loss='log',shuffle=True,learning_rate='adaptive', eta0=0.015, early_stopping=True, tol=1e-5, n_iter_no_change=100, max_iter=10000)
        elif classifierchoice == 'SGD5':
            classifier = SGDClassifier(loss='modified_huber',shuffle=True,learning_rate='adaptive', eta0=0.015, early_stopping=True, tol=1e-5, n_iter_no_change=100, max_iter=10000)
        elif classifierchoice == 'SVC':
            classifier = SVC(C=415.58623593356293, cache_size=300, class_weight='balanced', gamma=0.0008646611489221772, kernel='sigmoid', tol=0.00011171004269749147)
        elif classifierchoice == 'NNMLP':
            classifier = MLPClassifier(max_iter=300, learning_rate='adaptive', early_stopping=True, n_iter_no_change=30)
        elif classifierchoice == 'gradientboost':
            classifier = GradientBoostingClassifier(n_estimators=100) # Classifier
        elif classifierchoice == 'bagging':
            classifier = BaggingClassifier(base_estimator=DecisionTreeClassifier(), oob_score=True, random_state=101) #Bagging Classifier
        elif classifierchoice == 'linearreg':
            classifier = LogisticRegression()
        elif classifierchoice == 'linearreg2':
            classifier = LogisticRegression(solver='newton-cg',max_iter=2000, intercept_scaling=1.5,C=2.5,tol=1e-6)
        elif classifierchoice == 'linearreg3':
            classifier = LogisticRegression(solver='sag', max_iter=5000, intercept_scaling=0.5,C=5,tol=1e-7)
        elif classifierchoice == 'voting':
            classifier = VotingClassifier(estimators, voting='soft')
        
         #predicted Y
        if classifierchoice != 'voting':
            estimators.append((classifierchoice,classifier))
        classifier.fit(X_train_tf_NN, Y_train_NN)
        Y_pred = classifier.predict(X_test_tf)
        print(metrics.classification_report(Y_test, Y_pred, target_names=['negative', 'neutral']))

        print("Confusion Metrix: ")
        print(metrics.confusion_matrix(Y_test,Y_pred))

"""
FINDING A RIGHT CLASSIFIER
BINARY CLASSIFICATION OF NEGATIVE AND POSITIVE DATASET
FOR MULTICLASS CLASSIFICATION
"""
def classifierByPN():
    #TESTING
    #read Data
    test_csv = pd.read_csv('test_data_Processed.csv') #review rating
    test_csv_PN = test_csv.drop(test_csv[test_csv.Annotator_1 == 'neutral'].index)
    train_csv = pd.read_csv('train_data_Processed.csv') #sentiment
    train_csv_PN = train_csv.drop(train_csv[train_csv.sentiment == 'neutral'].index)
    train_csv_PN = train_csv_PN.sort_values('sentiment')
    train_csv_PN = train_csv_PN.drop(train_csv_PN[train_csv_PN.sentiment == 'positive'].index[:int(train_csv_PN.count()['sentiment']/1.4)])
    X_PN = train_csv_PN['review_content'] #column of reviews
    Y_PN = train_csv_PN['sentiment'] #Coresponding sentiment 
    X_test = test_csv_PN["review_content"]
    Y_test = test_csv_PN["Annotator_1"]
    X_train_PN = X_PN.values.astype('U')
    Y_train_PN = Y_PN
    X_test = X_test.values.astype('U')
    Y_test = Y_test.values.astype('U')
    tf_idf = TfidfVectorizer(sublinear_tf=True)
    #applying tf idf to training data
    X_train_tf_PN = tf_idf.fit_transform(X_train_PN) #creating all in tf-idf form
    X_train_tf_PN = tf_idf.transform(X_train_PN)
    
    #applying tf idf to test data
    X_test_tf = tf_idf.transform(X_test)
    print("(Test) n_samples: %d, n_features: %d" %X_test_tf.shape)
    print("(PN) n_samples: %d, n_features: %d" %X_train_tf_PN.shape)
    estimators = []

    while True:
        print('Input your classifier')
        classifierchoice = input()
        if classifierchoice == 'adaboost':
            classifier = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=100)
        elif classifierchoice == 'naive':
            classifier = MultinomialNB()
        elif classifierchoice == 'SGD2':
            classifier = SGDClassifier(loss='log',shuffle=True, warm_start=True)
        elif classifierchoice == 'SGD3':
            classifier = SGDClassifier(loss='modified_huber',shuffle=True)
        elif classifierchoice == 'SGD4':
            classifier = SGDClassifier(loss='log',shuffle=True,learning_rate='adaptive', eta0=0.01, early_stopping=True, tol=1e-4, n_iter_no_change=90, max_iter=5000)
        elif classifierchoice == 'SGD5':
            classifier = SGDClassifier(loss='modified_huber',shuffle=True,learning_rate='adaptive', eta0=0.01, early_stopping=True, tol=1e-4, n_iter_no_change=90, max_iter=5000)
        elif classifierchoice == 'decisiontree1':
            classifier = DecisionTreeClassifier()
        elif classifierchoice == 'decisiontree2':
            classifier = DecisionTreeClassifier(max_features='sqrt')
        elif classifierchoice == 'NNMLP':
            classifier = MLPClassifier(max_iter=300, learning_rate='adaptive', early_stopping=True, n_iter_no_change=30)
        elif classifierchoice == 'gradientboost':
            classifier = GradientBoostingClassifier(n_estimators=3000,warm_start=True,n_iter_no_change=50,learning_rate=0.08,tol=1e-5) # Classifier
        elif classifierchoice == 'bagging':
            classifier = BaggingClassifier(base_estimator=DecisionTreeClassifier(), oob_score=True, random_state=101) #Bagging Classifier
        elif classifierchoice == 'randomforest': 
            classifier = RandomForestClassifier(n_estimators=50,oob_score=True)
        elif classifierchoice == 'randomforest2':
            classifier = RandomForestClassifier(oob_score=True,warm_start=True)
        elif classifierchoice == 'linearreg':
            classifier = LogisticRegression()
        elif classifierchoice == 'linearreg2':
            classifier = LogisticRegression(solver='newton-cg',max_iter=2000, intercept_scaling=1.5,C=2.5,tol=1e-6)
        elif classifierchoice == 'linearreg3':
            classifier = LogisticRegression(solver='sag', max_iter=5000, intercept_scaling=0.5,C=5,tol=1e-7)
        elif classifierchoice == 'voting':
            classifier = VotingClassifier(estimators, voting='soft')
        
         #predicted Y
        if classifierchoice != 'voting':
            estimators.append((classifierchoice,classifier))
        classifier.fit(X_train_tf_PN, Y_train_PN)
        Y_pred = classifier.predict(X_test_tf)
        print(metrics.classification_report(Y_test, Y_pred, target_names=['neutral', 'positive']))

        print("Confusion Metrix: ")
        print(metrics.confusion_matrix(Y_test,Y_pred))

"""
FINDING A RIGHT CLASSIFIER
ENSEMBLE CLASSIFICATION 
"""
def classifierByEnsemble():
    #TESTING
    #read Data
    test_csv = pd.read_csv('test_data_Processed.csv') #review rating
    train_csv = pd.read_csv('train_data_Processed.csv') #sentiment
    train_csv = train_csv.sort_values('sentiment')
    train_csv = train_csv.drop(train_csv[train_csv.sentiment == 'positive'].index[int(train_csv.count()['sentiment']/5.5):])
    X = train_csv['review_content'] #column of reviews
    Y = train_csv['sentiment'] #Coresponding sentiment 
    X_test = test_csv["review_content"]
    Y_test = test_csv["Annotator_1"]
    X_train = X.values.astype('U')
    Y_train = Y
    X_test = X_test.values.astype('U')
    Y_test = Y_test.values.astype('U')
    tf_idf = TfidfVectorizer(sublinear_tf=True)
    #applying tf idf to training data
    X_train_tf = tf_idf.fit_transform(X_train) #creating all in tf-idf form
    X_train_tf = tf_idf.transform(X_train)
    #applying tf idf to test data
    X_test_tf = tf_idf.transform(X_test)
    print("n_samples: %d, n_features: %d" %X_test_tf.shape)
    print("n_samples: %d, n_features: %d" %X_train_tf.shape)
    #print(X_train_tf.toarray()) #One-hot vector of all words broken down
    estimators = []
    #classifier = MultinomialNB()  #Naive Bayes Classifier
    #estimators.append(('naivebayes',classifier))
    while True:
        print('Input your classifier')
        classifierchoice = input()
        if classifierchoice == 'OVR':
            classifier = OneVsOneClassifier(MLPClassifier(max_iter=300, learning_rate='adaptive', early_stopping=True, n_iter_no_change=30))
        elif classifierchoice == 'SVC':
            classifier = SVC(cache_size=400, class_weight='balanced', gamma='scale', kernel='sigmoid',probability=True,)
        elif classifierchoice == 'SVC3':
            classifier = SVC(C=415.58623593356293, cache_size=300, class_weight='balanced', probability=True, gamma=0.0008646611489221772, tol=0.00011171004269749147)
        elif classifierchoice == 'SVC4':
            classifier = SVC(cache_size=400, class_weight='balanced', gamma='scale', probability=True)
        elif classifierchoice == 'NNMLP':
            classifier = MLPClassifier(max_iter=500, learning_rate='adaptive', early_stopping=True, n_iter_no_change=30,)
        elif classifierchoice == 'logreg':
            classifier = LogisticRegression(multi_class='ovr')
        elif classifierchoice == 'logreg2':
            classifier = OneVsRestClassifier(LogisticRegression(multi_class='ovr', solver='sag', max_iter=5000, intercept_scaling=0.5,C=5,tol=1e-7))
        elif classifierchoice == 'logreg3':
            classifier = LogisticRegression(multi_class='multinomial')
        elif classifierchoice == 'svc2':
            classifier = OneVsRestClassifier(SVC(gamma='scale', class_weight='balanced'))
        elif classifierchoice == 'Linsvc':
            classifier = LinearSVC(class_weight='balanced')            
        elif classifierchoice == 'OCC2':
            classifier = OutputCodeClassifier(LinearSVC(class_weight='balanced'))
        elif classifierchoice == 'voting':
            classifier = VotingClassifier(estimators, voting='soft')
            
        #predicted Y
        if classifierchoice != 'voting':
            estimators.append((classifierchoice,classifier))
        classifier.fit(X_train_tf, Y_train)
        print(Y_test)
        Y_pred = classifier.predict(X_test_tf)
        print(metrics.classification_report(Y_test, Y_pred, target_names=['negative', 'neutral', 'positive']))

        print("Confusion Metrix: ")
        print(metrics.confusion_matrix(Y_test,Y_pred))

def classifierByNNeutral(test):
    test_csv = pd.read_csv('test_data_Processed.csv') #review rating
    test_csv_NN = test_csv.drop(test_csv[test_csv.Annotator_1 == 'positive'].index)
    train_csv = pd.read_csv('train_data_Processed.csv') #sentiment
    train_csv_NNegative = train_csv.drop(train_csv[train_csv.sentiment == 'positive'].index)
    X_NN = train_csv_NNegative['review_content'] #column of reviews
    Y_NN = train_csv_NNegative['sentiment'] #Coresponding sentiment 
    X_test = test_csv_NN["review_content"]
    X_train_NN = X_NN.values.astype('U')
    X_test = X_test.values.astype('U')
    tf_idf = TfidfVectorizer(sublinear_tf=True)
    #applying tf idf to training data
    X_train_tf_NN = tf_idf.fit_transform(X_train_NN) #creating all in tf-idf form
    X_train_tf_NN = tf_idf.transform(X_train_NN)
    
    #applying tf idf to test data
    X_test_tf = tf_idf.transform(X_test)
    print("(Test) n_samples: %d, n_features: %d" %X_test_tf.shape)
    print("(NN) n_samples: %d, n_features: %d" %X_train_tf_NN.shape)
    # estimators = []

    # classifier = SGDClassifier(loss='log',shuffle=True, warm_start=True)
    # estimators.append(('log',classifier))
    # classifier = SGDClassifier(loss='modified_huber',shuffle=True)
    # estimators.append(('mod_huber',classifier))
    # classifier = SGDClassifier(loss='log',shuffle=True,learning_rate='adaptive', eta0=0.015, early_stopping=True, tol=1e-5, n_iter_no_change=100, max_iter=10000)
    # estimators.append(('log2',classifier))
    # classifier = SGDClassifier(loss='modified_huber',shuffle=True,learning_rate='adaptive', eta0=0.015, early_stopping=True, tol=1e-5, n_iter_no_change=100, max_iter=10000)
    # estimators.append(('mod_huber2',classifier))
    # classifier = MLPClassifier(max_iter=500, learning_rate='adaptive', early_stopping=True, n_iter_no_change=30)
    # estimators.append(('NNMLP',classifier))
    # classifier = LogisticRegression()
    # estimators.append(('logreg',classifier))
    # classifier = LogisticRegression(solver='newton-cg',max_iter=2000, intercept_scaling=1.5,C=2.5,tol=1e-6)
    # estimators.append(('logreg2',classifier))
    # classifier = LogisticRegression(solver='sag', max_iter=5000, intercept_scaling=0.5,C=5,tol=1e-7)
    # estimators.append(('logreg3',classifier))
    # classifier = VotingClassifier(estimators, voting='soft')
    # classifier.fit(X_train_tf_NN, Y_NN)

    #Save classifier to speed up calculation

    filename = 'TFIDFClassifierNN.sav'
    #pickle.dump(classifier, open(filename, 'wb'))
    classifier = pickle.load(open(filename, 'rb'))
    
    #doing prediction
    test_input = tf_idf.transform(test)
    result = classifier.predict(test_input)[0]
    return result


"""
BINARY CLASSIFICATION OF NEGATIVE AND POSITIVE / NEGATIVE AND NEUTRAL DATASET 
FOR MULTICLASS CLASSIFICATION
TESTING ON ACCURACY AND SPEED OF CLASSIFICATION
"""
def classifierByPNegative(test):
    test_csv = pd.read_csv('test_data_Processed.csv') #review rating
    test_csv_NN = test_csv.drop(test_csv[test_csv.Annotator_1 == 'neutral'].index)
    train_csv = pd.read_csv('train_data_Processed.csv') #sentiment
    train_csv_NNegative = train_csv.drop(train_csv[train_csv.sentiment == 'neutral'].index)
    train_csv_NNegative = train_csv_NNegative.sort_values('sentiment')
    train_csv_NNegative = train_csv_NNegative.drop(train_csv_NNegative[train_csv_NNegative.sentiment == 'positive'].index[:int(train_csv_NNegative.count()['sentiment']/1.7)])
    X_NN = train_csv_NNegative['review_content'] #column of reviews
    Y_NN = train_csv_NNegative['sentiment'] #Coresponding sentiment 
    X_test = test_csv_NN["review_content"]
    X_train_NN = X_NN.values.astype('U')
    X_test = X_test.values.astype('U')
    tf_idf = TfidfVectorizer(sublinear_tf=True)
    #applying tf idf to training data
    X_train_tf_NN = tf_idf.fit_transform(X_train_NN) #creating all in tf-idf form
    X_train_tf_NN = tf_idf.transform(X_train_NN)
    
    #applying tf idf to test data
    X_test_tf = tf_idf.transform(X_test)
    print("(Test) n_samples: %d, n_features: %d" %X_test_tf.shape)
    print("(PN) n_samples: %d, n_features: %d" %X_train_tf_NN.shape)

    """
    MODEL SAVED INTO A PICKLE FORM THEREFORE COMMENTED AWAY.

    """
    # estimators = []

    # classifier = SGDClassifier(loss='log',shuffle=True, warm_start=True)
    # estimators.append(('log',classifier))
    # classifier = SGDClassifier(loss='modified_huber',shuffle=True)
    # estimators.append(('mod_huber',classifier))
    # classifier = SGDClassifier(loss='log',shuffle=True,learning_rate='adaptive', eta0=0.015, early_stopping=True, tol=1e-5, n_iter_no_change=100, max_iter=10000)
    # estimators.append(('log2',classifier))
    # classifier = SGDClassifier(loss='modified_huber',shuffle=True,learning_rate='adaptive', eta0=0.015, early_stopping=True, tol=1e-5, n_iter_no_change=100, max_iter=10000)
    # estimators.append(('mod_huber2',classifier))
    # classifier = MLPClassifier(max_iter=600, learning_rate='adaptive', early_stopping=True, n_iter_no_change=30)
    # estimators.append(('NNMLP',classifier))
    # classifier = LogisticRegression()
    # estimators.append(('logreg',classifier))
    # classifier = LogisticRegression(solver='newton-cg',max_iter=2000, intercept_scaling=1.5,C=2.5,tol=1e-6)
    # estimators.append(('logreg2',classifier))
    # classifier = LogisticRegression(solver='sag', max_iter=5000, intercept_scaling=0.5,C=5,tol=1e-7)
    # estimators.append(('logreg3',classifier))
    # classifier = VotingClassifier(estimators, voting='soft')
    # classifier.fit(X_train_tf_NN, Y_NN)

    #Save classifier to speed up calculation

    filename = 'TFIDFClassifierPN.sav'
    #pickle.dump(classifier, open(filename, 'wb'))
    classifier = pickle.load(open(filename, 'rb'))
    
    
    #doing prediction
    test_input = tf_idf.transform(test)
    result = classifier.predict(test_input)[0]
    return result


"""
    
    ACTUAL CLASSIFIER BY TFIDF
    CLASSIFIER USED SVC

"""
def classifierByTFIDF(test):
    #Read Data
    #test_csv = pd.read_csv('test_data.csv') #review rating
    #train_csv = pd.read_csv('train_data.csv') #sentiment
    test_csv = pd.read_csv('test_data_Processed.csv') #review rating
    train_csv = pd.read_csv('train_data_Processed.csv') #sentiment
    train_csv = train_csv.sort_values('sentiment')
    train_csv = train_csv.drop(train_csv[train_csv.sentiment == 'positive'].index[int(train_csv.count()['sentiment']/5.5):])
    X = train_csv['review_content'] #column of reviews
    Y = train_csv['sentiment'] #Coresponding sentiment 
    X_test = test_csv["review_content"]
    testlist = []
    positivecount = 0
    negativecount = 0
    neutralcount = 0
    """
    COMMENTED AWAY AS PREPROCESSING IS ALREADY SAVED IN A CSV FILE TO REDUCE THE TIME/ EFFORT TO PREPROCESS DATA AGAIN.
    """
    #Preprocessed Done and saved to CSV
    # #stopword removal and lemmatization
    # stopwords = stopwords.words('english')
    # lemmatizer = WordNetLemmatizer()
    # #print(stopwords)
    # #print(train_csv['review_content'].head())

    # #train test split
    # X = train_csv['review_content'].values.astype('U') #column of reviews
    # Y = train_csv['sentiment'] #Coresponding sentiment 
    # test_X = test_csv["review_content"].values.astype('U')
    # X_Processed = []
    # X_Test_Processed = []

    # #text pre-processing
    # for i in range(0, len(X)):
    #     review = re.sub('[^a-zA-Z]', ' ', X[i])
    #     review = review.lower()
    #     review = review.split()
    #     review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords)] #filter all stopwords.
    #     review = ' '.join(review)
    #     X_Processed.append(review)
    # print("Length of X and Y : %d" %len(X_Processed))

    # #test text
    # for i in range(0, len(test_X)):
    #     review = re.sub('[^a-zA-Z]', ' ', test_X[i])
    #     review = review.lower()
    #     review = review.split()
    #     review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords)] #filter all stopwords.
    #     review = ' '.join(review)
    #     X_Test_Processed.append(review)
    # print("Length of X and Y : %d" %len(X_Test_Processed))

    ##Preprocessed saved results.
    # train_csv = pd.concat([train_csv['product_name'],pd.Series(X_Processed,name='review_content'),Y],axis=1)
    # train_csv.to_csv('train_data_Processed.csv',index_label=False, index=False, mode = 'w')
    # test_csv = pd.concat([test_csv['product_name'],pd.Series(X_Test_Processed,name='review_content'),test_csv['Annotator_1']],axis=1)
    # test_csv.to_csv('test_data_Processed.csv',index_label=False, index=False, mode = 'w')

    #Can be done with 
    #train_test_split() but I want to experience splitting myself.
    # X_train = X_Processed[:int(np.floor(len(X_Processed)*0.9))]
    # X_test = X_Processed[int(np.floor(len(X_Processed)*0.9)):]
    # Y_train = Y[:int(np.floor(len(Y)*0.9))]
    # Y_test = Y[int(np.floor(len(Y)*0.9)):]
    # Y_test_list = Y_test.tolist()
    # print("positive: %d, neutral: %d, negative: %d" %(Y_test_list.count("positive"),Y_test_list.count("neutral"),Y_test_list.count("negative")))
    X_train = X.values.astype('U')
    Y_train = Y
    X_test = X_test.values.astype('U')
    #tf idf
    tf_idf = TfidfVectorizer(sublinear_tf=True)
    #applying tf idf to training data
    X_train_tf = tf_idf.fit_transform(X_train) #creating all in tf-idf form
    X_train_tf = tf_idf.transform(X_train)
    #applying tf idf to test data
    X_test_tf = tf_idf.transform(X_test)

    print("n_samples: %d, n_features: %d" %X_test_tf.shape)
    print("n_samples: %d, n_features: %d" %X_train_tf.shape)

    
    """
    MODEL IS SAVED IN A PICKLE THEREFORE WE ONLY NEED TO LOAD.
    """
    #print(X_train_tf.toarray()) #One-hot vector of all words broken down
    #classifier = SVC(gamma='scale', class_weight='balanced')
    # smt = SMOTE(random_state=777, k_neighbors=4)
    # X_train_tf, Y_train = smt.fit_resample(X_train_tf, Y_train)
    #classifier.fit(X_train_tf, Y_train)
    filename = 'TFIDFClassifier.sav'
    #Save classifier to speed up calculation
    #pickle.dump(classifier, open(filename, 'wb'))
    classifier = pickle.load(open(filename, 'rb'))
    
    #predicted Y
    #Y_pred = classifier.predict(X_test_tf)
    #print(metrics.classification_report(Y_test, Y_pred, target_names=['negative', 'neutral', 'positive']))

    #print("Confusion Metrix: ")
    #print(metrics.confusion_matrix(Y_test,Y_pred))
    for i in test:    
        #doing prediction
        review = re.sub('[^a-zA-Z]', ' ', i)
        review = review.lower()
        review = review.split()
        review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords)] #filter all stopwords.
        review = [' '.join(review)]
        test_input = tf_idf.transform(review)
        result = classifier.predict(test_input)[0]
        testlist.append(str(result))
        if result == 'positive':
            positivecount = positivecount + 1
        elif result == 'negative':
            negativecount = negativecount + 1
        else:
             neutralcount = negativecount + 1
    # Y_test = pd.DataFrame(data = ['negative'])
    # Y_test = Y_test[0]
    # print(metrics.classification_report(Y_test, testlist, target_names=['negative', 'neutral', 'positive']))

    # print("Confusion Metrix: ")
    # print(metrics.confusion_matrix(Y_test,testlist))
    
    """
    For sentiment analysis on product

    """
    # if positivecount > negativecount and positivecount > neutralcount:
    #       return testlist, 'positive'
    # elif positivecount < negativecount and negativecount > neutralcount:
    #       return testlist, 'negative'
    # else: 
    #       return testlist, 'neutral'

    return testlist

"""
EXAMPLE 

"""

TFIDFVisualise()
#classifierByNN()
# starttime = time.time()
# printlist = classifierByTFIDF(["I don't really like the product and my family hates it."])
# endtime = time.time()
# print(printlist)
# print("Time taken : {}".format(endtime-starttime))