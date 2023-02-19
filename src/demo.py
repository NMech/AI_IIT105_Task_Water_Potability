# -*- coding: utf-8 -*-
import warnings
warnings.simplefilter(action="ignore")
import joblib
import os
import numpy as np
import pandas as pd
root_directory = os.getcwd() # Run from current directory
#%%
np.random.seed(42)
#%%
model_filepath      = rf"{root_directory}/models"
model_filename      = "Voting_Classifier.joblib"
preprocess_filename = "preprocess_pipeline.joblib"
separator           = ","
ColumnsUsed         = ["ph", "Hardness", "Solids", "Chloramines", "Sulfate", "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"]
X_test_demo = pd.DataFrame([[7.16, 183.089, 6743.346, 3.803, 277.599, 428.036, 9.799, 90.03, 3.88],
                            [8.12, 103.589, 540.0810, np.nan, np.nan, 528.125, np.nan, 75.85, 2.07]],columns=ColumnsUsed)
#%%
model_fileInp           = os.path.join(model_filepath, model_filename)
pipe_preprocess_fileInp = os.path.join(model_filepath, preprocess_filename)
#%%
model    = joblib.load(model_fileInp)
pipeline = joblib.load(pipe_preprocess_fileInp)
#%%
X_test_prepared = pipeline.transform(X_test_demo)
prediction_demo = model.predict(X_test_prepared)