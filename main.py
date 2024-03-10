from Dt import DecisionTree
from SVM import supperVektor
from KNN import KNearest
from tools import Tools
        #------------------------------------------------------------------------#
        #                                                                        #
        #   Choice{                                                              #
        #   0: Predict (as <class 'numpy.ndarray'>),                             #
        #   1:Acc(as <class 'numpy.float64'>),                                   #
        #   2:confussion matrix(as <class 'numpy.ndarray'>),                     #
        #   3: class_resport (as <class 'dict'>)                                 #
        #   Return False means u choose false choice}                            #
        #                                                                        #
        #    url: CSV OR DATASET URL( AS STR)                                    #
        #                                                                        #
        #------------------------------------------------------------------------#

        #---------------------------------How Report List-----------------------------------#
        #                           report_list = [{"ML_Name": Dict}]                       #
        #-----------------------------------------------------------------------------------#
url = "glass.csv"

dt = DecisionTree(url)
Svc = supperVektor(url)
Knear = KNearest(url)

dt_x_train, dt_x_test, dt_y_train, dt_y_test = dt.data_preprocces()
svc_x_train, svc_x_test, svc_y_train, svc_y_test = Svc.data_preprocces()
knn_x_train, knn_x_test, knn_y_train, knn_y_test = Knear.data_preprocces()

dt_report =dt.predict_report(3, dt_x_train, dt_x_test, dt_y_train, dt_y_test)
svm_report =Svc.predict_report(3, svc_x_train, svc_x_test, svc_y_train, svc_y_test)
knn_report =Knear.predict_report(3, knn_x_train, knn_x_test, knn_y_train, knn_y_test)


report_list = [{"Decision_Tree": dt_report}, {"SVM": svm_report}, {"KNN": knn_report}]
Tools.Acc_table(report_list)
Tools.macro_prec(report_list)
Tools.macro_recall(report_list)
Tools.macro_f1(report_list)
Tools.wei_prec(report_list)
Tools.wei_recall(report_list)
Tools.wei_f1(report_list)
