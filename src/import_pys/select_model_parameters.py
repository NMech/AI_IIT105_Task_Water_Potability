# -*- coding: utf-8 -*-
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm      import SVC
#%%
model_parameters = {
    "gdb_clf": # Gradient Boosting
    {
        "model" :GradientBoostingClassifier(random_state=42),
        "params":
        {
            "learning_rate":[0.01, 0.1, 1.0],
            "n_estimators" :[50, 100, 200],
            "max_depth"    :[3, 6, 8, 10, 12],
            "max_features" :["sqrt","log2"],
        }
    },
        
    "gnd_clf": # Random Forest
    {
        "model":RandomForestClassifier(random_state=42),
        "params":
        {
            "n_estimators":[50, 100, 200],
            "max_depth"    :[3, 6, 8, 10, 12],
            "max_features":["auto","sqrt","log2"],
        }
    },
        
    "adab_clf": # ADA Boost
        {
            "model" : AdaBoostClassifier(random_state=42),
            "params":
            {
                "n_estimators" :[10, 50, 100, 200],
                "learning_rate":[0.01, 0.1, 1.0, 3.0],
                "algorithm"    :["SAMME", "SAMME.R"]
            }
        },
            
    "rbf_svm": # Gaussian RBF SVM
        {
            "model" : SVC(kernel="rbf", random_state=42),
            "params":
            {
                "gamma" : ["scale", "auto"],
                "C"     : [0.1, 0.5, 1.0]
            }
        }         
}