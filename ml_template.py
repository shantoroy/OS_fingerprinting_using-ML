import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import joblib


def preprocess(dataset, x_iloc_list, y_iloc, testSize):
    # dataset = pd.read_csv(csv_file)
    X = dataset.iloc[:, x_iloc_list].values 
    y = dataset.iloc[:, y_iloc].values 

    # split into training and testing set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize, random_state = 0)

    # standardization of values
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test, y_train, y_test
    
    
class classification:
    
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
    
    # Contour Graph if no. of feature is 2
    def classification_view(self, X_train, y_train, classifier):
        from matplotlib.colors import ListedColormap
        X_set, y_set = X_train, y_train
        X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                             np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
        plt.figure(figsize=(16,8))
        plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                     alpha = 0.4, cmap = ListedColormap(('#F5716C', '#39A861')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                        c = ListedColormap(('#F5716C', '#39A861'))(i), label = j)
        plt.title('Visualization of how the classification is made: errors are identified where the color of the\
                                    point and the background are different')
        plt.legend()
        plt.show()
        

    def accuracy(self, confusion_matrix):
        sum, total = 0,0
        for i in range(len(confusion_matrix)):
            for j in range(len(confusion_matrix[0])):
                if i == j: 
                    sum += confusion_matrix[i,j]
                total += confusion_matrix[i,j]
        return sum/total
    
    
    def LR(self):
        from sklearn.linear_model import LogisticRegression
        lr_classifier = LogisticRegression()
        lr_classifier.fit(self.X_train, self.y_train)
        joblib.dump(lr_classifier, "model/lr.sav")
        y_pred = lr_classifier.predict(self.X_test)

        print("\n")
        print("### Logistic Regression Classifier ###")
        print('Classification Report: ')
        print(classification_report(self.y_test, y_pred),'\n')
        print('Confusion Matrix: ')
        print(confusion_matrix(self.y_test, y_pred),'\n')
        print('Precision: ', self.accuracy(confusion_matrix(self.y_test, y_pred))*100,'%')
        
        if len(self.X_train[0]) == 2:
            self.classification_view(self.X_train, self.y_train, lr_classifier)
        
        
    def KNN(self):
        from sklearn.neighbors import KNeighborsClassifier
        knn_classifier = KNeighborsClassifier()
        knn_classifier.fit(self.X_train, self.y_train)
        joblib.dump(knn_classifier, "model/knn.sav")
        y_pred = knn_classifier.predict(self.X_test)
        
        print("\n")
        print("### K-Neighbors Classifier ###")
        print('Classification Report: ')
        print(classification_report(self.y_test, y_pred),'\n')
        print('Confusion Matrix: ')
        print(confusion_matrix(self.y_test, y_pred),'\n')
        print('Precision: ', self.accuracy(confusion_matrix(self.y_test, y_pred))*100,'%')
        
        if len(self.X_train[0]) == 2:
            self.classification_view(self.X_train, self.y_train, knn_classifier)
        
    
    # kernel type could be 'linear' or 'rbf' (Gaussian)
    def SVM(self, kernel_type):
        from sklearn.svm import SVC
        svm_classifier = SVC(kernel = kernel_type)
        svm_classifier.fit(self.X_train, self.y_train)
        joblib.dump(svm_classifier, "model/svm.sav")
        y_pred = svm_classifier.predict(self.X_test)
        
        print("\n")
        print("### Support Vector Classifier (" + kernel_type + ") ###")
        print('Classification Report: ')
        print(classification_report(self.y_test, y_pred),'\n')
        print('Confusion Matrix: ')
        print(confusion_matrix(self.y_test, y_pred),'\n')
        print('Precision: ', self.accuracy(confusion_matrix(self.y_test, y_pred))*100,'%')
        
        if len(self.X_train[0]) == 2:
            self.classification_view(self.X_train, self.y_train, svm_classifier)
        
        
    def NB(self):
        from sklearn.naive_bayes import GaussianNB
        nb_classifier = GaussianNB()
        nb_classifier.fit(self.X_train, self.y_train)
        joblib.dump(nb_classifier, "model/nb.sav")
        y_pred = nb_classifier.predict(self.X_test)
        
        print("\n")
        print("### Naive Bayes Classifier ###")
        print('Classification Report: ')
        print(classification_report(self.y_test, y_pred),'\n')
        print('Confusion Matrix: ')
        print(confusion_matrix(self.y_test, y_pred),'\n')
        print('Precision: ', self.accuracy(confusion_matrix(self.y_test, y_pred))*100,'%')
        
        if len(self.X_train[0]) == 2:
            self.classification_view(self.X_train, self.y_train, nb_classifier)
        
        
    def DT(self):
        from sklearn.tree import DecisionTreeClassifier
        tree_classifier = DecisionTreeClassifier()
        tree_classifier.fit(self.X_train, self.y_train)
        joblib.dump(tree_classifier, "model/tree.sav")
        y_pred = tree_classifier.predict(self.X_test)
        
        print("\n")
        print("### Decision Tree Classifier ###")
        print('Classification Report: ')
        print(classification_report(self.y_test, y_pred),'\n')
        print('Confusion Matrix: ')
        print(confusion_matrix(self.y_test, y_pred),'\n')
        print('Precision: ', self.accuracy(confusion_matrix(self.y_test, y_pred))*100,'%')
        
        if len(self.X_train[0]) == 2:
            self.classification_view(self.X_train, self.y_train, tree_classifier)
        
        
    def RF(self):
        from sklearn.ensemble import RandomForestClassifier
        rf_classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')
        rf_classifier.fit(self.X_train, self.y_train)
        joblib.dump(rf_classifier, "model/rf.sav")
        y_pred = rf_classifier.predict(self.X_test)
        
        print("\n")
        print("### Random Forest Classifier ###")
        print('Classification Report: ')
        print(classification_report(self.y_test, y_pred),'\n')
        print('Confusion Matrix: ')
        print(confusion_matrix(self.y_test, y_pred),'\n')
        print('Precision: ', self.accuracy(confusion_matrix(self.y_test, y_pred))*100,'%')
        
        if len(self.X_train[0]) == 2:
            self.classification_view(self.X_train, self.y_train, rf_classifier)
            
            
