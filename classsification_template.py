import pandas as pd
import numpy as np
from scipy.sparse.construct import random

class Classification():
    def __init__(self, x, y):
        self.x = x.values
        self.y = y.values
        self.x_train = 0
        self.y_train = 0
        self.x_test = 0
        self.y_test = 0
        self.y_pred = 0
        self.check = False

    def split_datas(self, test_size):
        # SPLIT DATA
        from sklearn.model_selection import train_test_split
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size = test_size, random_state=0)
        # SCALING PROCESS
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        self.x_train = sc.fit_transform(self.x_train)
        self.x_test = sc.fit_transform(self.x_test)
        return self.x_train, self.x_test, self.y_train, self.y_test
    def check_implemantation(self) : 
        if(self.check != False) :
            self.x_train = 0
            self.y_train = 0
            self.x_test = 0
            self.y_test = 0
            self.y_pred = 0
        else :
            pass
    def accuracy(self):
        from sklearn.metrics import confusion_matrix, accuracy_score
        cm = confusion_matrix(self.y_test, self.y_pred)
        ac = accuracy_score(self.y_test, self.y_pred)
        print("CONFUSION MATRIX:\n" , cm)
        print("ACCURACY SCORE:\n" , ac)
    
    # CLASSIFIERS
    def logistic_reg(self, test_size) :
        # CHECK ANY CLASSIFICATION BE IMPLEMENTED BEFORE 
        self.check_implemantation()
        # IMPORT AND SPLIT THE DATA
        from sklearn.linear_model import LogisticRegression
        self.x_train, self.x_test, self.y_train, self.y_test = self.split_datas(test_size)
        classifier = LogisticRegression(random_state=0)
        # FIT AND PREDICTION
        classifier.fit(self.x_train, self.y_train)
        self.y_pred = classifier.predict(self.x_test)
        self.check = True
    
    def KNN(self, test_size):
        # CHECK ANY CLASSIFICATION BE IMPLEMENTED BEFORE 
        self.check_implemantation()
        # IMPORT AND SPLIT THE DATA
        from sklearn.neighbors import KNeighborsClassifier 
        self.x_train, self.x_test, self.y_train, self.y_test = self.split_datas(test_size)
        classifier = KNeighborsClassifier(n_neighbors = 5, metric= 'minkowski', p = 2)
        # FIT AND PREDICTION
        classifier.fit(self.x_train, self.y_train)
        self.y_pred = classifier.predict(self.x_test)
        self.check = True

    def SVM(self, test_size, kernel) : 
        # CHECK ANY CLASSIFICATION BE IMPLEMENTED BEFORE 
        self.check_implemantation()
        # IMPORT AND SPLIT THE DATA
        from sklearn.svm import SVC
        self.x_train, self.x_test, self.y_train, self.y_test = self.split_datas(test_size)
        classifier = SVC(kernel= kernel, random_state=0)
        # FIT AND PREDICTION
        classifier.fit(self.x_train, self.y_train)
        self.y_pred = classifier.predict(self.x_test)
        self.check = True

    def naive_bayes(self, test_size) : 
        # CHECK ANY CLASSIFICATION BE IMPLEMENTED BEFORE 
        self.check_implemantation()
        # IMPORT AND SPLIT THE DATA
        from sklearn.naive_bayes import GaussianNB
        self.x_train, self.x_test, self.y_train, self.y_test = self.split_datas(test_size)
        classifier = GaussianNB()
        # FIT AND PREDICTION
        classifier.fit(self.x_train, self.y_train)
        self.y_pred = classifier.predict(self.x_test)
        self.check = True

    def decision_tree_classifier(self, test_size):
        # CHECK ANY CLASSIFICATION BE IMPLEMENTED BEFORE 
        self.check_implemantation()
        # IMPORT AND SPLIT THE DATA
        from sklearn.tree import DecisionTreeClassifier
        self.x_train, self.x_test, self.y_train, self.y_test = self.split_datas(test_size)
        classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
        # FIT AND PREDICTION
        classifier.fit(self.x_train, self.y_train)
        self.y_pred = classifier.predict(self.x_test)
        self.check = True

    def random_forest_classifier(self, test_size) : 
        # CHECK ANY CLASSIFICATION BE IMPLEMENTED BEFORE 
        self.check_implemantation()
        # IMPORT AND SPLIT THE DATA
        from sklearn.ensemble import RandomForestClassifier
        self.x_train, self.x_test, self.y_train, self.y_test = self.split_datas(test_size)
        classifier = RandomForestClassifier(criterion='entropy', random_state=0, n_estimators=30)
        # FIT AND PREDICTION
        classifier.fit(self.x_train, self.y_train)
        self.y_pred = classifier.predict(self.x_test)
        self.check = True
    
# READ DATA
data = pd.read_csv("Data.csv")
x = data.iloc[:,:-1]
y = data.iloc[:,-1]

# CREATE MODEl
classifier = Classification(x, y)

# LOGISTIC REGRESSION
classifier.logistic_reg(0.25)
print("LOGISTIC REGRESSION SCORE:")
classifier.accuracy()

# K-NEIGBORS-CLASSIFICATION
print("--------------------------------")
classifier.KNN(0.25)
print("KNN SCORE:")
classifier.accuracy()
    
# SUPPORT-VECTOR-MACHINE
print("--------------------------------")
classifier.SVM(0.25, kernel = 'rbf')
print("SVM SCORE:")
classifier.accuracy()

# NAIVE-BAYES
print("--------------------------------")
classifier.naive_bayes(0.25)
print("NAIVE-BAYES SCORE:")
classifier.accuracy()

# DTC
print("--------------------------------")
classifier.decision_tree_classifier(0.25)
print("DECISION TREE SCORE:")
classifier.accuracy()

# RFC
print("--------------------------------")
classifier.random_forest_classifier(0.25)
print("RANDOM FOREST SCORE:")
classifier.accuracy()