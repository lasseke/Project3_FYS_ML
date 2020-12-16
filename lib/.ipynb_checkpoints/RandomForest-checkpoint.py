'''
Custom implementation of the Random Forest algorithm (Breimann 2001). A customized number of decision trees with a specified depth are built using bootstrap samples of the input data.
In the end, majority voting of all trees defines the classification or regression result.
'''

### Import dependency packages ###
import numpy as np


'''
Metrics functions unique for RandomForest
'''
#import Evaluation


###################################################################################################
################################################ Random Forest ####################################
###################################################################################################
'''
Random forest class
'''

class RandomForest(object):
    
    '''
    Constructor. Input argument variables:
    
    max_depth = max. levels between root and leaf node. 0 means no limitation.
    mtry = amount of input features to randomly draw from full set that are used for splitting. 
    Default is sqrt of total number of input features.
    task = 'classification' or 'regression'
    split_method = 'gini' or 'entropy'
    num_trees = number of trees that are constructed and that will vote in the end
    verbose = '1' for spam, '0' for quiet
    '''
    def __init__(self, num_trees=500, max_depth=0, mtry=0, task='classification',\
                 split_method='gini', verbose=True):
        
        self.num_trees = abs(int(num_trees))
        self.max_depth = abs(int(max_depth))
        self.mtry = abs(int(mtry))
        if task not in ['classification','regression']:
            raise ValueError("Task must be 'classification' or 'regression'!")
        self.task = task
        if split_method is not 'gini':# or 'entropy':
            raise ValueError("Split method can currently only be 'gini'!")
        self.split_method = split_method
        self.verbose = bool(verbose)
        
        ### Additionally needed variables ###
        self.variable_names = [] # String list of feature variable names
        self.target_names = [] # String list of target (class) names
        self.tree_list = [] # List to store grown tree objects
        
        self.n_features = 0 # Amount of Features
        self.n_targets = 0 # Amount of targets
        self.n_obs = 0 # Amount of observations
        
        self.X = None # Feature matrix used to grow the forest
        self.y = None # Corresponding targets
        
    
    '''
    Define standard output string for printing a RandomForest object
    '''
    def __str__(self):
        return "Random Forest for " + self.task + " (#Vars: " + str(self.n_features) + ", #Classes: " + \
        str(self.n_targets) + "). #Trees: " + str(self.num_trees) + ", Max. depth: " + \
        str(self.max_depth) + ", mtry: " + str(self.mtry) + "."
    
    

    '''
    Fit the random forest to input data! Also prints progress if verbose == True.
    X = feature matrix
    y = target class
    features = list of variable names
    '''
    def growForest(self, X, y, feature_names=None, target_names=None):
        
        ### Import bootstrap libraries? ###
        #from sklearn.utils import resample
        
        ################################### Check input #####################################
        if X.shape[0] != len(y):
            return "Error - number of rows in feature matrix X and targets in y not identical!"
        
        if type(y) is np.ndarray and y.ndim != 1:
            return "Error - y must be a 1 dimensional array!"
        
        if y is not list:
            y = list(y)
            
        if feature_names is not None:
            if len(feature_names) != X.shape[1]:
                return "Error - feature name list length and feature matrix column amount must be identical!"
        
        if target_names is not None:
            if len(target_names) != len(set(y)):
                return "Error - target name list length and amount of unique target vector classes must be identical!"
        
        ################################### Generate names ##################################
        
        if feature_names is None:
            feature_names = ["feat_"+str(x+1) for x in range(X.shape[1])]
        
        
        if target_names is None:
            target_names = ["target_"+str(x+1) for x in range(len(y))]
        
        
        ################################### Set parameters ##################################
        
        ### Set mtry to square root of number of features if not specified in constructor ###
        if self.mtry == 0:
            self.mtry = int(np.sqrt(X.shape[1]))
        
        ### FEATURE NAMES ###
        self.variable_names = feature_names
        self.n_features = len(feature_names)
        
        ### TARGET NAMES ###
        self.target_names = target_names
        self.n_targets = len(target_names)

        
        
        ### NUMBER OF OBS ###
        self.n_obs = X.shape[0]
        
        ### Store X and y ###
        self.X = X
        self.y = y
    
        
        ################################# Loop through trees ################################
        
        ### Print verbose! Import process bar
        if self.verbose:   
            from tqdm import tqdm
            treeIter = tqdm(range(self.num_trees))
        ### Otherwise no progress bar
        else:
            treeIter = range(self.num_trees)
        
        ### Loop through number of trees...
        for i in treeIter:

            ### Make new bootstrap sample
            bs_idx = np.random.choice(range(self.n_obs), size = self.n_obs, replace=True) # Bootstrap indices for rows in X/y
            # Out of bag indices for samples not contained in boot
            oob_idx = [i for i in range(self.n_obs) if i not in bs_idx] 
            
            # Set bootstrap and oob samples
            X_boot= X[bs_idx, :]
            y_boot = [y[i] for i in bs_idx]
            
            ### Randomly choose current split variables based on amount specified in mtry ###
            cur_mtry_indices = sorted(np.random.choice(range(self.n_features), size = self.mtry, replace=False))
            
            ### Create new tree
            cur_tree = Tree(max_d = self.max_depth, split_m = self.split_method)
            # Grow tree using only randomly selected vars
            
            cur_tree.fit(X_boot[:,cur_mtry_indices], y_boot, oob_idx = oob_idx,\
                         feat_names = [feature_names[i] for i in cur_mtry_indices], target_names=target_names)
            
            ### Append tree to tree list!
            self.tree_list.append(cur_tree)
                
        ### Finished
        return "Random Forest succesfully grown :-)"
    
    
    
    ################################################ Predict unseen data ##############################################
    
    
    '''
    Make predictions for new data (=not used for growing the forest). Input matrix must have same
    number of columns and order as used for training.
    A name can be provided in "_id" to store a list of votes of individual trees for a specific prediction.
    Note: to also allow string predictions, this method returns a list object. Convert to numpy array if needed.
    '''
    def predict(self, X):
        
        n_trees = len(self.tree_list)
        n_obs = X.shape[0]
        
        if n_trees == 0:
            return "Error - forest not grown yet!"
        
        ### Store tree votes in a list
        vote_list = []
        predicted_y = []
        

        # Save all current votes in list
        cur_vote_list = []

        # Loop through all trees in forest...
        for i,_tree in enumerate(self.tree_list):
            # ...and get their indivdual predictions
            cur_vars = _tree.getFeatNames()
            cur_varidx = [idx for idx,val in enumerate(self.variable_names) if val in cur_vars] # Get indices of vars that were used for this tree

            cur_vote = _tree.predict(X[:,cur_varidx])
            # Add to list
            cur_vote_list.append(cur_vote)
            print('Tree ' + str(i+1) + ' of ' + str(n_trees) + ' has voted.', end='\r')
        
        print("",end="\n")
        ### After each tree has voted, determine most frequent value and add it as prediction
        predicted_y = []
        
        for i in range(n_obs):
            cur_list = []
            for j in range(len(cur_vote_list)):
                cur_list.append(cur_vote_list[j][i]) # Add vote of current tree for current observation to a list
            
            # Determine most frequent class, add to predictions
            predicted_y.append(RandomForest.mostFrequent(cur_list))
            
        #final_vote = RandomForest.mostFrequent(cur_vote_list)
        #predicted_y.append(final_vote)#(max(set(cur_vote_list), key = cur_vote_list.count) )
        ### Also save full vote list for further analysis
        #vote_list.append(cur_vote_list)
        
        ### Return
        return predicted_y, cur_vote_list
    
    
    ################################################ Out of Bag Error calculation ##############################################
    
    '''
    Calculate Out of bag error!
    '''
    def calcOobError(self):
        
        y_predict_list = []
        
        # Loop through all indices
        for obs_idx in range(self.n_obs):
            
            tree_idx_list = [] # Store the indices of trees that did not contain current observation
            
            for tree_idx, _tree in enumerate(self.tree_list):
                if _tree.wasOutOfBag(obs_idx):
                    tree_idx_list.append(tree_idx)
            
            ### Use resulting trees to make prediction for current sample!
            cur_vote_list = []
            
            # Loop through relevant trees
            for i in tree_idx_list:
                # Get current trees configuration, i.e. used variables and corresponding indices
                cur_vars = self.tree_list[i].getFeatNames()
                cur_varidx = [idx for idx,val in enumerate(self.variable_names) if val in cur_vars]
                # Make prediction
                cur_vote = self.tree_list[i].predict(self.X[obs_idx,cur_varidx])
                cur_vote_list.append(cur_vote)
            
            
            ### After each tree has voted the current sample, determine most frequent value and add it as prediction
            final_vote = RandomForest.mostFrequent(cur_vote_list)
            y_predict_list.append(final_vote)
    
            
        import lib.Evaluation as ev
        ### Calculate accuracy, oob error = 1-Accuracy
        # Convert y to string
        #y_str = [str(s) for s in self.y]
        acc = ev.accuracy(self.y, y_predict_list)
        print("Accuracy: "+str(acc))
        oob_err = 1.0 - acc
        print("Oob error: "+str(oob_err))
            
        return oob_err
            
    
    
    ###########################################
    ########## Get and set functions ##########
    ###########################################
    
    def getMaxDepth(self):
        return self.max_depth
    
    @staticmethod
    def mostFrequent(_list): 
        
        if type(_list) is not list:
            raise ValueError("Must pass an object of type list!")
        
        try:
          if len(_list) < 1:
            raise ValueError("List does not contain anything!")
        except:
          print("List empty!")
        
        ## Use count functionality to find most frequent list element
        _mode = None
        counter = 0
        
        for cur_val in _list:
            cur_count = _list.count(cur_val)
            if cur_count > counter or counter == 0:
                _mode = cur_val
                counter = cur_count
        
        return _mode

    
###################################################################################################
################################################ CART Tree ########################################
###################################################################################################

'''
Class to represent indivdual decision trees grown in a forest.
'''
class Tree(object):

    '''
    Constructor. Provide:
    X, y, indices of variables to use in X (optional), max depth, split method.
    '''
    def __init__(self, max_d=None, split_m="gini"):
            
        self.max_d = max_d
        self.split_m = split_m
        
        ### Additional variables ###
        self.depth = 0
        self.n_classes_tree = 0
        self.n_features_tree = 0
        self.oobIdx = None
        
        ### Names of features and targets
        self.feat_names, self.target_names = None, None
    
    ################################################ Growing the tree ##############################################
    
    '''
    Fit function that recursively calls growTree() and stores oob indices of data
    '''
    def fit(self, X, y, oob_idx=None, feat_names=None, target_names=None, method="gini"):
        
        ### Store out of bag indices, if provided
        self.oob_idx = oob_idx
        
        self.feat_names, self.target_names = feat_names, target_names
        
        ### Store amount of unique classes and features for whole tree
        self.classes_tree = sorted(set(y))
        self.n_features_tree = X.shape[1]
        
        self.nodes = self.growTree(X,y)
        
        
    '''
    Function to recursively grow a CART.
    '''
    def growTree(self, X, y, depth=0):
        
        ### Class occurrences in current node
        num_samples_per_class = [np.sum(y == _class) for _class in self.classes_tree]
        
        # Predicted class is the one with most occurrences in y.
        predicted_class = self.classes_tree[np.argmax(num_samples_per_class)]
        
        ### Create first node
        node = Node(gini_idx=self.calcGiniIndex(y), num_samples = len(y),\
                    num_samples_per_class = num_samples_per_class, predicted_class = predicted_class)
        
        ### RECURSIVE SPLIT UNTIL MAX DEPTH REACHED OR PERFECT GINIS ###
        if depth < self.max_d and node.getGiniIdx() != 0.0:
            # Make a new split
            idx, thr, bestgini = self.makeSplit(X, y)
            
            # Test if set can be splitted further
            if idx is not None:
                
                indices_left = X[:, idx] < thr  # Store all values smaller than the threshold that end up in left node...
                X_left, y_left = X[indices_left], [i for (i, v) in zip(y, indices_left) if v] # Add them to new matrices...
                X_right, y_right = X[~indices_left], [i for (i, v) in zip(y, indices_left) if not v] # Add remaining observations (larger than thresh.) to right node
                
                node.setFeatureIdx(idx) # Store index for best feature in node
                node.setThreshold(thr) # Store threshold value for split in node
                
                ### Create new nodes with subset of data
                node.left_node = self.growTree(X_left, y_left, depth + 1)
                node.right_node = self.growTree(X_right, y_right, depth + 1)
        
        ### In final iteration: return the root node, that now links all other nodes until leaves
        # Before: currently new grown node
        return node
        
    '''
    Method to make a best split of data between nodes
    '''
    def makeSplit(self, X, y):
        
        ### Save amount of observations
        n_obs = len(y)
        
        ### Need more than one observation
        if n_obs <= 1:
            return None, None
        
        ### Save current unique classes in list type, also allows cat./str. vars
        parent_classes = sorted(set(y))
        n_classes = len(parent_classes)
        # Save amount of indiv. class occurrences for parent node
        class_freq_list_parent = [np.sum(y == _class) for _class in parent_classes] 
        
        
        ### Calculate parent node Gini index
        best_gini = self.calcGiniIndex(y)
        
        ### Store index of feature that gives best split and the corresponding threshold value
        best_idx, best_thresh = None, None
        
        
        ############################### LOOP THROUGH FEATURES ################################
        for feat_idx in range(X.shape[1]):
            
            ### Sort data according to currently examined feature, reduces complexity ###
            sort_idx = np.argsort( X[:, feat_idx] )
            
            X_sorted = X[sort_idx, feat_idx]
            y_sorted = [y[i] for i in sort_idx]
            
            ### Class frequencies for potential child nodes ###
            class_freq_list_left = np.zeros_like(class_freq_list_parent) ### Left node class freq. zeros in the beginning
            class_freq_list_right = class_freq_list_parent.copy() ### Begin right node with same class freq. as parent
            
            
            ############### LOOP THROUGH OBSERVATIONS ###############
            for i in range(1, n_obs):
                
                ### Determine class of current observation from sorted vector
                cur_class = y_sorted[i-1] # i starts at 1, hence decrease
                
                ### Decrease cur. occurrence from right node (in the beginning containing all samples) and add to left
                cur_class_idx = parent_classes.index(cur_class) # Implemented as list type
                
                class_freq_list_left[cur_class_idx] += 1
                class_freq_list_right[cur_class_idx] -= 1
                
                ### Calculate new Gini indices for left and right nodes ###
                # i represents amount of obs moved to left node and hence n_obs_left #
                gini_left = 1.0 - sum(
                    (class_freq_list_left[x] / i) ** 2 for x in range(n_classes)
                )
                gini_right = 1.0 - sum(
                    (class_freq_list_right[x] / (n_obs - i)) ** 2 for x in range(n_classes)
                )
                
                
                ### Calculate overall new gini index as the weighted average of the Gini impurity of the children.
                gini = (i * gini_left + (n_obs - i) * gini_right) / n_obs
                
                
                ### The following condition is to make sure we don't assign two identical values to different groups,
                ### i.e. identical observations have to end up in the same node
                if X_sorted[i] == X_sorted[i-1]:
                    continue
                    
                ### Update parameters if better Gini is found! ###
                if gini < best_gini:
                    best_gini = gini
                    best_idx = feat_idx
                    best_thresh = (X_sorted[i] + X_sorted[i - 1]) / 2  # Define threshold as median of the values
          
        ### Return best values!
        #print("best idx: "+str(best_idx)+", best thresh: "+str(best_thresh)+", best gini: "+str(best_gini))
        return best_idx, best_thresh, best_gini

    
    
    ################################################ Making predictions ##############################################

        
    '''
    Make a prediction on fresh data with the grown tree.
    '''
    def predict(self, X):
        
        ### Get linked nodes for tree
        node = self.nodes
        
        if node is None:
            return "Error - tree not grown yet!"
        
        ### Was a list passed? Convert to numpy.ndarray
        if type(X) is list:
            X = np.array(X)
          
        ### Was a vector passed? Slightly adjust the code
        if X.ndim == 1:
            
            if len(X) != self.n_features_tree:
                raise ValueError("Input feature dimensions must match data used to grow tree!")
        
            # Store prediction
            cur_prediction = node.getPredictedClass()
            
            # Value at Feature (given as index) that splits data best in current node based on training
            cur_value = X[node.getFeatureIdx()] 
            
            ### UPDATE VALUES (i.e. move down nodes) UNTIL TERMINAL NODE IS REACHED ###
            while(node.getNextNode(cur_value)):
                
                node = node.getNextNode(cur_value)         # Set currently linked node to child node
                cur_prediction = node.getPredictedClass()  # Store predicted class of new node (overwrite old pred.)
                cur_value = X[node.getFeatureIdx()]      # Get value of observation matrix at position of best feature of new node
                ### Restart loop with new value
                
            ### When while loop finished, add final prediction to list
            y_predicted = cur_prediction
        
            # Finally, return list of predictions
            return y_predicted
        
        ############################################################################################################################
        # In case of matrix
        
        if X.shape[1] != self.n_features_tree:
            raise ValueError("Input feature dimensions must match data used to grow tree!")
        
        ### Initialize list to store predictions
        y_predicted = []
        
        ### Loop through each observations, i.e. row in matrix...
        for i in range(X.shape[0]):
            
            # Set back to initial node
            cur_node = node
            
            # Store prediction
            cur_prediction = cur_node.getPredictedClass()
            
            # Value at Feature (given as index) that splits data best in current node based on training
            cur_value = X[i, cur_node.getFeatureIdx()] 
            #print(cur_value)
            #print(cur_node.getThreshold())
            ### UPDATE VALUES (i.e. move down nodes) UNTIL TERMINAL NODE IS REACHED ###
            while(cur_node.getNextNode(cur_value)):
                
                cur_node = cur_node.getNextNode(cur_value)         # Set currently linked node to child node
                cur_prediction = cur_node.getPredictedClass()  # Store predicted class of new node (overwrite old pred.)
                cur_value = X[i, cur_node.getFeatureIdx()]      # Get value of observation matrix at position of best feature of new node
                ### Restart loop with new value
                
            ### When while loop finished, add final prediction to list
            y_predicted.append(cur_prediction)
        
        # Finally, return list of predictions
        return y_predicted
                
        
    ################################################ Additional functionality ##############################################

    '''
    Miscelaneous functions.
    '''
    
    ### Test if a sample (_int = its row index) was used to grow this tree. Needed for Out of bag error calculation. 
    def wasOutOfBag(self,_int):
        # Simple test
        if _int in self.oob_idx:
            return True
        else:
            return False
        
    def getFeatNames(self):
        return self.feat_names
    
    def getTargetNames(self):
        return self.target_names
    
    ###########################################
    ### Static methods to calculate metrics ###
    ###########################################
    
    '''
    Static method to calculate the Gini index.
    '''
    @staticmethod
    def calcGiniIndex(y):
        
        ### Store unique classes and length of n (# of obs)
        classes = set(y)
        n = len(y)
        y = list(y)
        
        ### Gini index = 1 - sum of individual fractions of class occurences squared
        gini_classes = [(y.count(_class)/n)**2 for _class in classes]
        
        gini_index = 1 - np.sum(gini_classes)
        
        return gini_index
    
            


###################################################################################################
################################################ CART Node ########################################
###################################################################################################

'''
Class to represent indivdual Nodes in a tree.
'''
class Node(object):
        
    def __init__(self, gini_idx, num_samples, num_samples_per_class, predicted_class):

        # Initialize class instances
        self.gini_idx = gini_idx # This nodes gini index
        self.num_samples = num_samples # Number of samples in node
        self.num_samples_per_class = num_samples_per_class # Class frequencies in node
        self.predicted_class = predicted_class
        
        ### Additional variables
        # Store left and right nodes
        self.left_node = None
        self.right_node = None
        
        self.featureIdx = None # Index of feature in original data frame
        self.threshold = None # Split value. Cont: Smaller --> left_node, larger --> right_node
        
    
    '''
    Get and set methods
    '''
    def setFeatureIdx(self, idx):
        self.featureIdx = idx
        return True
        
    def setThreshold(self, thresh):
        self.threshold = thresh
        return True
        
    def getFeatureIdx(self):
        return self.featureIdx
    
    def getThreshold(self):
        return self.threshold
    
    def getPredictedClass(self):
        return self.predicted_class
    
    def getGiniIdx(self):
        return self.gini_idx

    '''
    Return following node based on value of given variable
    '''
    def getNextNode(self, value):
        
        if self.getThreshold() is None:
            return None
        elif value < self.getThreshold() and self.left_node is not None: # Smaller then threshold --> left node
            return self.left_node
        elif self.right_node is not None:
            return self.right_node
        else:
            return None # Will finally return None when terminal node is reached
         