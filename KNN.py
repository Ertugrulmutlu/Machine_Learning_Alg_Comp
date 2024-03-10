import pandas as pd
import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import classification_report, confusion_matrix

class KNearest:
    def __init__(self,csv_url):
        self.url = csv_url
    def data_preprocces(self):

        #------------------------------------------------------------#
        #                                                            #
        #   This code return x_train, x_test, y_train, y_test .      #
        #   That means its retrun test and train datas.              #
        #------------------------------------------------------------#

        data = pd.read_csv(self.url, sep=",") #Get Csv file and sep by Koma
        columns  = data.columns #Get column names
        X = np.array(data.drop([columns[len(columns)-1]], axis=1))  #Get x data which is  Features
        y = np.array(data[columns[len(columns)-1]]) #Get y data which is we want predict
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y, test_size= 0.2) #train test data split

        return x_train, x_test, y_train, y_test

    def predict_report(self,choice,  x_train, x_test, y_train, y_test, Neighbor_c = 9):

        #------------------------------------------------------------------------#
        #                                                                        #
        #   0: Predict (as <class 'numpy.ndarray'>),                             #
        #   1:Acc(as <class 'numpy.float64'>),                                   #
        #   2:confussion matrix(as <class 'numpy.ndarray'>),                     #
        #   3: class_resport (as <class 'dict'>)                                 #
        #   Return False means u choose false choice                             #
        #                                                                        #
        #------------------------------------------------------------------------#

        try:
            choice = int(choice)
        except:
            print("bir hata oluştu")
            return False  
        KNN = KNeighborsClassifier(n_neighbors=Neighbor_c)
        KNN.fit(x_train,y_train)    
        predic = KNN.predict(x_test)
        if choice == 0:
            return predic
        elif choice ==1:
            acc = KNN.score(x_test,y_test)
            return acc
        elif choice ==2:
            conf_mat = confusion_matrix(y_test, predic)
            return conf_mat
        elif choice ==3:
            class_rep = classification_report(y_test, predic, output_dict=True)
            return class_rep
        else:
            return False


