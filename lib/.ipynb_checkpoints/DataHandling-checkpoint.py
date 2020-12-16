'''
Miscellaneous code for data handling and testing.
'''

def getIrisData(test_fraction=None):
    
    ### Import iris data set ###
    from sklearn import datasets
    
    ### Load Iris data set
    iris = datasets.load_iris()
    X = iris.data#[:, :2]  # we only take the first two features.
    y = iris.target

    X_labs = ["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width"]
    y_labs = ["Setosa", "Versicolour", "Virginica"]
    
    if test_fraction is None:
        return X, y, X_labs, y_labs
    else:
        # Import split package
        from sklearn.model_selection import train_test_split
        ### TRAIN TEST SPLITS
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_fraction)

        return X_train, X_test, y_train, y_test, X_labs, y_labs
    