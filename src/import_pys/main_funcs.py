# -*- coding: utf-8 -*-
import time
import numpy as np
from sklearn.model_selection import cross_val_score
#%%
def Classifier_Fit(classifier, X_train, y_train, Cv, clf_txt):
    """
    Implementation of classifier fitting.
    """
    t1 = time.time()
    classifier.fit(X_train, y_train)
    classifier_CV_score = cross_val_score(classifier, X_train, y_train, scoring="accuracy", cv=Cv, n_jobs=-1)
    score = round(np.mean(classifier_CV_score),3)
    t2 = time.time()
    msg = f"{clf_txt:30} {round(t2-t1,4)}s"
    print(msg)
    return classifier, score


def return_q1_q3(df):
    """
    Valid only for this exercise (not fgeneralised).
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
        
    