import matplotlib.pyplot as plt

class Tools:
    def __init__(self):
        pass
    def Acc_table(List_of_keys: list):
        x= []
        y= []
        #------------XandY-----------#
        for i in List_of_keys:
            Ml_alg_name= list(i.keys())[0] 
            x.append(Ml_alg_name)
        #------------XandY-----------#
        for i in List_of_keys:
            for key in x:
                try:
                    y.append(i[key]["accuracy"])
                    print(key, ">>", i[key]["accuracy"])
                    break
                except:
                    pass
        fig, ax = plt.subplots()
        ax.stem(x, y)
        plt.title("accuracy")
        plt.show()
    def macro_prec(List_of_keys: list):
        x= []
        y= []
        #------------XandY-----------#
        for i in List_of_keys:
            Ml_alg_name= list(i.keys())[0] 
            x.append(Ml_alg_name)
        #------------XandY-----------#
        for i in List_of_keys:
            for key in x:
                try:
                    y.append(i[key]["macro avg"]["precision"])
                    print(key, ">>", i[key]["macro avg"]["precision"])
                    break
                except:
                    pass
        fig, ax = plt.subplots()
        ax.stem(x, y)
        plt.title("Macro Avg ---> Precision")
        plt.show()   

    def macro_recall(List_of_keys: list):
        x= []
        y= []
        #------------XandY-----------#
        for i in List_of_keys:
            Ml_alg_name= list(i.keys())[0] 
            x.append(Ml_alg_name)
        #------------XandY-----------#
        for i in List_of_keys:
            for key in x:
                try:
                    y.append(i[key]["macro avg"]["recall"])
                    print(key, ">>", i[key]["macro avg"]["recall"])
                    break
                except:
                    pass
        fig, ax = plt.subplots()
        ax.stem(x, y)
        plt.title("Macro Avg ---> Recall")
        plt.show()  
    def macro_f1(List_of_keys: list):
            x= []
            y= []
            #------------XandY-----------#
            for i in List_of_keys:
                Ml_alg_name= list(i.keys())[0] 
                x.append(Ml_alg_name)
            #------------XandY-----------#
            for i in List_of_keys:
                for key in x:
                    try:
                        y.append(i[key]["macro avg"]["f1-score"])
                        print(key, ">>", i[key]["macro avg"]["f1-score"])
                        break
                    except:
                        pass
            fig, ax = plt.subplots()
            ax.stem(x, y)
            plt.title("Macro Avg ---> F1 Score")
            plt.show()  
    def macro_sup(List_of_keys: list):
            x= []
            y= []
            #------------XandY-----------#
            for i in List_of_keys:
                Ml_alg_name= list(i.keys())[0] 
                x.append(Ml_alg_name)
            #------------XandY-----------#
            for i in List_of_keys:
                for key in x:
                    try:
                        y.append(i[key]["macro avg"]["support"])
                        print(key, ">>", i[key]["macro avg"]["support"])
                        break
                    except:
                        pass
            fig, ax = plt.subplots()
            ax.stem(x, y)
            plt.title("Macro Avg ---> Support")
            plt.show()  

    def wei_prec(List_of_keys: list):
        x= []
        y= []
        #------------XandY-----------#
        for i in List_of_keys:
            Ml_alg_name= list(i.keys())[0] 
            x.append(Ml_alg_name)
        #------------XandY-----------#
        for i in List_of_keys:
            for key in x:
                try:
                    y.append(i[key]["weighted avg"]["precision"])
                    print(key, ">>", i[key]["weighted avg"]["precision"])
                    break
                except:
                    pass
        fig, ax = plt.subplots()
        ax.stem(x, y)
        plt.title("weighted Avg ---> Precision")
        plt.show()   

    def wei_recall(List_of_keys: list):
        x= []
        y= []
        #------------XandY-----------#
        for i in List_of_keys:
            Ml_alg_name= list(i.keys())[0] 
            x.append(Ml_alg_name)
        #------------XandY-----------#
        for i in List_of_keys:
            for key in x:
                try:
                    y.append(i[key]["weighted avg"]["recall"])
                    print(key, ">>", i[key]["weighted avg"]["recall"])
                    break
                except:
                    pass
        fig, ax = plt.subplots()
        ax.stem(x, y)
        plt.title("weighted Avg ---> Recall")
        plt.show()  
    def wei_f1(List_of_keys: list):
            x= []
            y= []
            #------------XandY-----------#
            for i in List_of_keys:
                Ml_alg_name= list(i.keys())[0] 
                x.append(Ml_alg_name)
            #------------XandY-----------#
            for i in List_of_keys:
                for key in x:
                    try:
                        y.append(i[key]["weighted avg"]["f1-score"])
                        print(key, ">>", i[key]["weighted avg"]["f1-score"])
                        break
                    except:
                        pass
            fig, ax = plt.subplots()
            ax.stem(x, y)
            plt.title("weighted Avg ---> F1 Score")
            plt.show()  
    def wei_sup(List_of_keys: list):
            x= []
            y= []
            #------------XandY-----------#
            for i in List_of_keys:
                Ml_alg_name= list(i.keys())[0] 
                x.append(Ml_alg_name)
            #------------XandY-----------#
            for i in List_of_keys:
                for key in x:
                    try:
                        y.append(i[key]["weighted avg"]["support"])
                        print(key, ">>", i[key]["weighted avg"]["support"])
                        break
                    except:
                        pass
            fig, ax = plt.subplots()
            ax.stem(x, y)
            plt.title("weighted Avg ---> Support")
            plt.show()  