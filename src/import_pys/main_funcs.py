# -*- coding: utf-8 -*-
import os
import pickle
import time
import numpy as np
from   sklearn.model_selection import cross_val_score
#%%
def Classifier_Fit(classifier, X_train, y_train, Cv, clf_txt, filepathOut):
    """
    Implementation of classifier fitting.\n
    Keyword arguments:\n
        classifier : sklearn classifier object.\n
        X_train    : Training dataset [pd.DataFrame].\n
        y_train    : Labels of training dataset [pd.Series].\n 
        Cv         : Number of cross-validation folds [int].\n
        clf_txt    : Text to print in screen (also used for naming the model).\n
        filepathOut: Filepath where the models are saved.\n
    """
    t1 = time.time()
    classifier.fit(X_train, y_train)
    classifier_CV_score = cross_val_score(classifier, X_train, y_train, scoring="accuracy", cv=Cv, n_jobs=-1)
    score = round(np.mean(classifier_CV_score),3)
    Save_Classifier(classifier, filepathOut, clf_txt)
    t2 = time.time()
    msg = f"{clf_txt:40} |{round(t2-t1,4)}s"
    print(msg)

    return classifier, score

def return_q1_q3(df):
    """
    Valid only for this exercise (not generalised).\n
    Keyword arguments:\n
        df : pd.DataFrame.\n
    """
    df    = df.copy()
    q1_q3 = {}
    for col in df.columns[:-1]:
        q1_q3 = df[col].quantile([0.25, 0.5, 0.75])
        
        q1 = q1_q3 [0.25]
        q3 = q1_q3 [0.75]
        median     = q1_q3 [0.5]
        IQR        = q3 - q1
        Lower_fence = q1 - IQR
        Upper_fence = q3 + IQR
                      
        df.loc[df[col]>=Upper_fence, col] = median
        df.loc[df[col]<=Lower_fence, col] = median
        
    return df

def Save_Classifier(classifier, filepathOut, filename):
    """
    Used for saving the trained model using pickle.\n
    """
    fileOut = os.path.join(filepathOut, filename)
    pickle.dump(classifier, open(fileOut, "wb"))
    
    return None