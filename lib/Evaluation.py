'''
Add python files exclusive for Evaluation here!
'''
### Import packages!
import numpy as np
import pandas as pd

'''
Calculate the total accuracy, i.e. fraction of correctly predicted classes
'''
def accuracy(y, y_predict):
    
    y, y_predict = __checkObsPredInput__(y, y_predict)
    
    ### Store variables
    n = len(y)
    correct = 0
    
    for i in range(n):
        if y[i] == y_predict[i]:
            correct += 1
    
    return (correct/n)
    
'''
Create a confusion matrix for multi class classification. Return pandas DataFrame.
'''

def confMatMetrics(y, y_predict, class_dict=None):
    
    # Check input
    y, y_predict = __checkObsPredInput__(y, y_predict)
    
    ### Suppress pandas warning, desired behavior
    pd.options.mode.chained_assignment = None  # default='warn'
    
    # Number of unique variables
    set_y, set_y_predict = sorted(set(y)), sorted(set(y_predict))
    merged = set_y + set_y_predict
    set_merged = sorted(set(merged))
    n = len(set_merged)
    
    # Use dictionary if none provided!
    d_keys = range(n)    
    d_values = set_merged
    
    ### Create dictionary out of key value pairs
    if class_dict is None:
        
        # Create confusion matrix
        mat = pd.DataFrame(np.zeros(shape=(n,n),dtype=int), columns = d_values, index=d_values)
        # Fill matrix
        for i,cur_obs in enumerate(y):
            cur_pred = y_predict[i]
            mat[cur_pred][cur_obs] += 1
    
    else:
        
        if type(class_dict) is not dict:
            raise ValueError("'class_dict' must be 'dict' type!")
        # Add found variables to list
        cur_vars = [val for val in d_values if val in list(class_dict.keys())]
        
        if len(class_dict) != len(d_values):
            raise ValueError("Dictionary length and provided class vector length must match!")
            
        # Create confusion matrix
        mat = pd.DataFrame(np.zeros(shape=(n,n),dtype=int), columns = list(class_dict.values()), index=list(class_dict.values()))
        
        # Fill matrix
        for i,cur_obs in enumerate(y):
            cur_pred = y_predict[i]
            cur_pred = class_dict.get(cur_pred)
            cur_obs = class_dict.get(cur_obs)
            mat[cur_pred][cur_obs] += 1
    
    
    ### Calculate indices
    met_names = ['Accuracy','Precision','Recall','F1_score','n_obs']
    overall_metrics = pd.DataFrame(np.zeros(shape=(n+1,len(met_names))), columns = met_names, index = list(class_dict.values()) + ["Total_OR_WeightedAvg"])
    
    ### First, calculate overall Accuracy
    overall_metrics['Accuracy'] = 'NA'
    overall_metrics['Accuracy']["Total_OR_WeightedAvg"] = np.sum(np.diagonal(mat)) / np.sum(np.sum(mat))
    
    
    ### Loop through the confusion matrix classes
    for _class in list(class_dict.values()):

        ### Calculate
        cur_precision = mat[_class][_class] / (np.sum(mat[_class]))
        cur_recall = mat[_class][_class] / (np.sum(mat.loc[_class]))

        overall_metrics['Precision'][_class] = cur_precision
        overall_metrics['Recall'][_class] = cur_recall
        overall_metrics['F1_score'][_class] = 2* (cur_precision * cur_recall) / (cur_precision + cur_recall)
        overall_metrics['n_obs'][_class] = np.sum(mat.loc[_class])
        
    ### Calculate final values
    total_n = np.sum(overall_metrics['n_obs'])
    overall_metrics['n_obs']['Total_OR_WeightedAvg'] = total_n
    
    # Weighted Precision/Recall/F1
    overall_metrics['F1_score']['Total_OR_WeightedAvg'] = np.sum([n*val for n,val in tuple(zip(overall_metrics['F1_score'],overall_metrics['n_obs']))]) / total_n
    overall_metrics['Precision']['Total_OR_WeightedAvg'] = np.sum([n*val for n,val in tuple(zip(overall_metrics['Precision'],overall_metrics['n_obs']))]) / total_n
    overall_metrics['Recall']['Total_OR_WeightedAvg'] = np.sum([n*val for n,val in tuple(zip(overall_metrics['Recall'],overall_metrics['n_obs']))]) / total_n
    
    # Convert n to int for readability
    overall_metrics['n_obs'] = [int(n) for n in overall_metrics['n_obs']]
    
    ## Round values where appropriate
    overall_metrics['Precision'] = [np.round(x,3) for x in overall_metrics['Precision']]
    overall_metrics['Recall'] = [np.round(x,3) for x in overall_metrics['Recall']]
    overall_metrics['F1_score'] = [np.round(x,3) for x in overall_metrics['F1_score']]
    
    
    overall_metrics['Accuracy']["Total_OR_WeightedAvg"] = np.round(overall_metrics['Accuracy']["Total_OR_WeightedAvg"],3)
    
    ### RETURN HERE ###
    return mat, overall_metrics
    

'''
Internal methods for testing input validity
'''   
def __checkObsPredInput__(y, y_predict):
    
    if type(y) is int or type(y) is float or type(y_predict) is int or type(y_predict) is float:
        raise ValueError("Must provide a list or 1D-array!")
    
    if len(y) != len(y_predict):
        raise ValueError("Predictions and observations must have same length!")
    
    y, y_predict = [str(x) for x in y], [str(x) for x in y_predict]
    
    return y, y_predict