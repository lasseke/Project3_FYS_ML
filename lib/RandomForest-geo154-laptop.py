'''
Custom implementation of the Random Forest algorithm (Breimann 2001). A customized number of decision trees with a specified depth are built using bootstrap samples of the input data.
In the end, majority voting of all trees defines the classification or regression result.
'''

### Functions needed here!




'''
Random forest class
'''

class RandomForest(object):
    
    '''
    Constructor. Input argument variables:
    
    max_depth = max. levels between root and leaf node. 0 means no limitation.
    mtry = amount of input features to randomly draw from full set that are used for splitting. Usually: sqrt of total number of input features.
    task = 'classification' or 'regression'
    split_method = 'gini' or 'entropy'
    num_trees = number of trees that are constructed and that will vote in the end
    verbose = '1' for spam, '0' for quiet
    '''
    def __init__(self,\
                 num_trees=500, max_depth=0, mtry=6, task='classification', split_method='gini', verbose=True):
        
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.mtry = mtry
        self.task = task
        self.split_method = split_method
        self.verbose = verbose
        self.variable_list = []
        self.class_list = []
        self.tree_list = []
        
    
    '''
    Define standard output string for printing a RandomForest object
    '''
    def __str__(self):
        return "Random Forest for " + self.task + " (#Vars: " + len(self.variable_list) + ", #Classes: " + len(self.class_list) +\
        "). #Trees: " + self.num_trees + ", Max. depth: " + self.max_depth + ", mtry: " + self.mtry "."
    
    
    ################################################### Get and Add functions ##########################################################
    
    def getMaxDepth(self):
        return self.max_depth
    
    
    ################################################### Inner class for CARTs #########################################################
    
    '''
    Inner class for individual decision trees
    '''
    class Tree(object):
        
        def __init__(self, parent, X, y):
            
            #growTree()
            self.splits = []
            self.parent = parent # Save reference to parent RF object
        
        def growTree(self):
            
            self.depth = 0
            max_depth = parent.getMaxDepth()
            
            while self.depth <= max_depth:
                pass
            
            return 0

            
        def vote(self, X):
            return prediction
        
        
    ################################################### Fitting and predicting ########################################################    
        
    '''
    Fit the random forest to input data! Also prints progress if verbose == True.
    '''
    def grow(self,X,y):
        
        ### Import bootstrap libraries
        from sklearn.utils import resample
        
        if self.verbose:   
            from tqdm import tqdm
            #from time import sleep
        
            for i in tqdm(range(self.num_trees)):
            #sleep(0.3)
                
                ### Make bootstrap sample
                X_boot, y_boot = resample(X, y, random_state=0)
                
                self.tree_list.append(Tree(self, X_boot, y_boot))
            pass
            
        
        return 0
    
    
    ##################################################### Static methods ############################################################
    
    '''
    Static method to calculate the Gini index
    '''
    @staticmethod
    def getGiniIndex(X, target, split_var = None):
        
        ### Calculate gini index when no splitting variable is provided
        if split_var is None:
            base_prob = 0
        
        ### Calculate Gini index
        gini_index=0
        for values in set(X): # Loop through unique values or categories of input vector
            gini_value=gini(df,feature,category,target,classes_list) 
            P_k_a_value=P_k_a(category)
            gini_index+=gini_value*P_k_a_value
        
        return gini_index
    
    
    '''
    Static method to perform a split based on desired variable and threshold
    '''
    @staticmethod
    def makeSplit(X, classes, thresh = None):
        
        ### Calculate gini index when no splitting variable is provided
        if split_var is None:
            base_prob = 0
        
        return
        
        
        
        
        