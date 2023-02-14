# -*- coding: utf-8 -*-
import warnings
warnings.simplefilter(action="ignore")
#%%
import os
import sys
import time
import pandas as pd
import numpy  as np
import json
import random
from   tabulate import tabulate
#%%
root_directory = os.getcwd() # Run from current directory
sys.path.append("./import_pys")
sys.path.append("./import_pys/Plots")
from data_preprocessing            import Pipeline_preprocessor
from CorrelationMatrixPlot         import CorrelationMatrixPlot
from MissingValuesPlot             import MissingValuesPlot
from FeaturesHistogramPlot         import FeaturesHistogramPlot
from ClassificationMetricsPlot     import ClassificationMetricsPlot
from BoxPlot                       import boxplot_potability
from select_model_parameters       import model_parameters
from main_funcs                    import Classifier_Fit, return_q1_q3, Save_Classifier
#%%
from sklearn.model_selection       import train_test_split, GridSearchCV
from sklearn.metrics               import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing         import PolynomialFeatures
from sklearn.linear_model          import LogisticRegression, RidgeClassifier, Perceptron
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.gaussian_process      import GaussianProcessClassifier
from sklearn.naive_bayes           import GaussianNB
from sklearn.svm                   import SVC, LinearSVC
from sklearn.neighbors             import KNeighborsClassifier
from sklearn.ensemble              import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
#%%
np.random.seed(42)
#%%
###############################################################################
################################## Input data #################################
###############################################################################
data_filepath = rf"{root_directory}/import_data"
fig_filepath  = rf"{root_directory}/../Report/Figures"
res_filepath  = rf"{root_directory}/Results"
model_filepath= rf"{root_directory}/models"
data_filename = "water_potability.csv"
separator     = ","
plotDiagrams      = True
saveDiagrams      = True
RunModelSelection = False # True only for choosing the final model hyperparameters (Time consuming)
replaceOutliers   = False 
testSize  = 0.2
Cv        = 5 # cross-validation folds
nFeatures = "all" # How many/which features to include in training. Options: "all"/integer (# of random features)
#%%
###############################################################################
######################### Creation of plotting objects ########################
###############################################################################
missingPlotObj = MissingValuesPlot() 
histPlotObj    = FeaturesHistogramPlot()
#%%
###############################################################################
############################ Read data & get a look ###########################
###############################################################################
fileInp   = os.path.join(data_filepath, data_filename)
data      = pd.read_csv(fileInp, sep=separator)
datadescr = data.describe()
datadescrPrint = tabulate(datadescr,headers=datadescr.columns, tablefmt= "grid")
with open(rf"{res_filepath}/water_potability_descr.dat","w") as fileOut:
    fileOut.write(datadescrPrint)
    
# In case of nFeatures -> int select nFeatures random features
Columns = list(data.columns)
if nFeatures == "all": 
    col_idx = "all"
    ColumnsUsed = Columns[:-1]#exclude last column (label)
elif type(nFeatures) == int and (nFeatures < 9 and nFeatures > 2):
    col_idx = sorted(random.sample(range(9), nFeatures))
    ColumnsUsed = [Columns[idx] for idx in col_idx]
else:
    raise Exception("Exception!!! Check nFeatures input")
    
if plotDiagrams == True:
    missingPlotObj.MatrixPlot(data, savePlot=[saveDiagrams, fig_filepath, "Missing_Values_Matrix"])
    missingPlotObj.BarPlot(data, savePlot=[saveDiagrams, fig_filepath, "Missing_Values_Bar"])
#%%
###############################################################################
####################### Data cleaning & transformations #######################
###############################################################################
pipeline = Pipeline_preprocessor([], ColumnsUsed)
X        = pipeline.fit_transform(data[ColumnsUsed])
X        = pd.DataFrame(X, columns=ColumnsUsed)
y        = data.Potability
XYconcat = pd.concat([X,y], axis=1) # DataFrame used in training procedure
if plotDiagrams == True:
    histPlotObj.HistPlot(XYconcat, "Potability", savePlot=[saveDiagrams, fig_filepath, "Features_Histogram"])
#%%
###############################################################################
########################### Correlation Coefficients ##########################
###############################################################################
pearsonCorr  = XYconcat.corr(method="pearson")
spearmanCorr = XYconcat.corr(method="spearman")
if plotDiagrams == True:
    corrPlot = CorrelationMatrixPlot()
    corrPlot.PlotCorrelationHeatMaps(pearsonCorr, colorMap="mySymmetric", Title="Pearson Coefficient", Rotations=[0.,45.], savePlot=[saveDiagrams,fig_filepath,"Pearson_coeff"],showtriL=True)
    corrPlot.PlotCorrelationHeatMaps(spearmanCorr, colorMap="mySymmetric", Title="Spearman Coefficient", Rotations=[0.,45.], savePlot=[saveDiagrams,fig_filepath,"Spearman_coeff"],showtriL=True)
#%%
###############################################################################
############################## Outlier detection ##############################
###############################################################################
if plotDiagrams == True:
    boxplot_potability(data, savePlot=[saveDiagrams, fig_filepath, "boxplot"])
if replaceOutliers == True:
    data_potable = return_q1_q3(data[data["Potability"]==0])
    data_nonpotable = return_q1_q3(data[data["Potability"]==1])
    data = pd.concat((data_potable, data_nonpotable))
#%%
###############################################################################
########################### Split dataset train-test ##########################
###############################################################################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, shuffle=True, random_state=42)
#%%
#-----------------------------------------------------------------------------#
###############################################################################
######################## Testing different classifiers ########################
###############################################################################
models_CV_res = []  
#%%
###############################################################################
############################### Ridge classifier ##############################
###############################################################################
ridge_clf = RidgeClassifier(alpha=0.5)
clf_txt   = "1 Ridge Regression"
ridge_clf, score = Classifier_Fit(ridge_clf, X_train, y_train, Cv, clf_txt, model_filepath)
models_CV_res.append([clf_txt, score]) 
#%%
###############################################################################
######################## Logistic Regression classifier #######################
###############################################################################
log_clf = LogisticRegression(random_state=0, solver='liblinear')
clf_txt = "2 Logistic Regression"
log_clf, score = Classifier_Fit(log_clf, X_train, y_train, Cv, clf_txt, model_filepath)
models_CV_res.append([clf_txt, score]) 
#%%
###############################################################################
############################## Linear Perceptron ##############################
###############################################################################
prc_clf = Perceptron()
clf_txt = "3 Linear Perceptron"
prc_clf, score = Classifier_Fit(prc_clf, X_train, y_train, Cv, clf_txt, model_filepath)
models_CV_res.append([clf_txt, score]) 
#%%
###############################################################################
######################### Linear Discriminant Analysis ########################
###############################################################################
lda_clf = LinearDiscriminantAnalysis()
clf_txt = "4 Linear Discriminant Analysis"
lda_clf, score = Classifier_Fit(lda_clf, X_train, y_train, Cv, clf_txt, model_filepath)
models_CV_res.append([clf_txt, score]) 
#%%
###############################################################################
####################### Quadratic Discriminant Analysis #######################
###############################################################################
qda_clf = QuadraticDiscriminantAnalysis()
clf_txt = "5 Quadratic Discriminant Analysis"
qda_clf, score = Classifier_Fit(qda_clf, X_train, y_train, Cv, clf_txt, model_filepath)
models_CV_res.append([clf_txt, score]) 
#%%
###############################################################################
######################### Gaussian Process Classifier #########################
###############################################################################
gpc_clf = GaussianProcessClassifier()
clf_txt = "6 Gaussian Process Classifier"
gpc_clf, score = Classifier_Fit(gpc_clf, X_train, y_train, Cv, clf_txt, model_filepath)
models_CV_res.append([clf_txt, score]) 
#%%
###############################################################################
############################# Gaussian Naive Bayes ############################
###############################################################################
NB_clf  = GaussianNB()
clf_txt = "7 Gaussian Naive Bayes"
NB_clf, score = Classifier_Fit(NB_clf, X_train, y_train, Cv, clf_txt, model_filepath)
models_CV_res.append([clf_txt, score]) 
#%%
###############################################################################
############################# Linear SVM classifier ###########################
###############################################################################
svm_clf = LinearSVC(C=1, loss="hinge")
clf_txt = "8 Linear SVM"
svm_clf, score = Classifier_Fit(svm_clf, X_train, y_train, Cv, clf_txt, model_filepath)
models_CV_res.append([clf_txt, score]) 
#%%
###############################################################################
################################ Polynomial SVM ###############################
###############################################################################
pol  = PolynomialFeatures(degree=3)
Xpol = pol.fit_transform(X_train)
pol_svm_clf = LinearSVC(C=1, loss="hinge")
clf_txt     = "9 Polynomial SVM"
pol_svm_clf, score = Classifier_Fit(pol_svm_clf, X_train, y_train, Cv, clf_txt, model_filepath)
models_CV_res.append([clf_txt, score]) 
#%%
###############################################################################
########################### Gaussian RBF SVM kernel ###########################
###############################################################################
rbf_kernel_svm_clf = SVC(kernel="rbf",gamma="scale",C=1.0)
clf_txt            = "10 Gaussian RBF SVM"
rbf_kernel_svm_clf, score = Classifier_Fit(rbf_kernel_svm_clf, X_train, y_train, Cv, clf_txt, model_filepath)
models_CV_res.append([clf_txt, score]) 
#%%
###############################################################################
############################ k-Neighbors classifier ###########################
###############################################################################
kNeighbor_clf  = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
clf_txt = "11 k-Neighbors"
kNeighbor_clf, score = Classifier_Fit(kNeighbor_clf, X_train, y_train, Cv, clf_txt, model_filepath)
models_CV_res.append([clf_txt, score]) 
#%%
###############################################################################
################################ Random Forest ################################
###############################################################################
rnd_clf = RandomForestClassifier(n_estimators=200, max_features="sqrt", max_depth=10, max_leaf_nodes=None, n_jobs=-1)
clf_txt = "12 Random Forest"
rnd_clf, score = Classifier_Fit(rnd_clf, X_train, y_train, Cv, clf_txt, model_filepath)
models_CV_res.append([clf_txt, score]) 
feature_importance = rnd_clf.feature_importances_
#%%
###############################################################################
############################## Gradient Boosting ##############################
###############################################################################
gdb_clf = GradientBoostingClassifier()
clf_txt = "13 Gradient Boosting"
Classifier_Fit(gdb_clf, X_train, y_train, Cv, clf_txt, model_filepath)
models_CV_res.append([clf_txt, score]) 
#%%
###############################################################################
################################## Ada Boost ################################## 
###############################################################################
ada_clf = AdaBoostClassifier()
clf_txt = "14 Ada Boost"
Classifier_Fit(ada_clf, X_train, y_train, Cv, clf_txt, model_filepath)
models_CV_res.append([clf_txt, score]) 
#%%
models_CV_res_tab = tabulate(models_CV_res, tablefmt='grid')
with open(rf"{res_filepath}/models_CV_res_replaceOutliers_{bool(replaceOutliers)}_Features_{col_idx}.dat","w") as fileOut:
    fileOut.write(models_CV_res_tab)
#%%
###############################################################################
############################### Model Selection ###############################
###############################################################################
if RunModelSelection == True:
    t1 = time.time()
    msg = "######## Executing models hyperparameters selection ########"
    print(msg)
    cv_scores = {}
    for model_name, params in model_parameters.items():
        grid_search = GridSearchCV(params["model"], params["params"], cv=Cv)
        grid_search.fit(X,y)
        cv_scores[model_name]  = [grid_search.best_params_ , grid_search.best_score_]
        models_hyperparameters = cv_scores
        models_hyperparameters_df = pd.DataFrame.from_dict(cv_scores, orient="index", columns=["hyper_params", "Best_score"])
        
        grid_res = tabulate(pd.DataFrame(grid_search.cv_results_) , headers="keys", tablefmt='grid')  
        with open(rf"{res_filepath}/grid_res_{model_name}.dat","w") as fileOut:
            fileOut.write(grid_res)

    models_hyperparameters_txt = tabulate(models_hyperparameters, tablefmt="grid")
    json_object = json.dumps(cv_scores, indent=4)
    with open(rf"{res_filepath}/models_hyperparams.json", "w") as outfile:
        outfile.write(json_object)
        
    t2 = time.time()
    msg = f"Model Selection {round((t2-t1)/60,2)}mins"
    msg +="\n######### End of models hyperparameters selection #########"
    print(msg)
else:
    with open(rf"{res_filepath}/models_hyperparams.json", "r") as inpfile:
        models_hyperparameters = json.load(inpfile)
#%%
###############################################################################
############################## Create final model #############################
###############################################################################
gdb_clf = GradientBoostingClassifier(**models_hyperparameters["gdb_clf"][0], random_state=42)
Save_Classifier(gdb_clf, model_filepath, "Gradient_Boosting_optimized")
gnd_clf = RandomForestClassifier(**models_hyperparameters["gnd_clf"][0], random_state=42)
Save_Classifier(gnd_clf, model_filepath, "Random_Forest_optimized")
rbf_svm = SVC(**models_hyperparameters["rbf_svm"][0], kernel="rbf", random_state=42)
Save_Classifier(rbf_svm,  model_filepath, "RBF_SVM_optimized")
final_model_clf = VotingClassifier(estimators=[
                                     ("gdb_clf",  gdb_clf),
                                     ("gnd_clf",  gnd_clf),
                                     ("rbf_svm",  rbf_svm),
                                    ], voting="hard")  
Save_Classifier(final_model_clf, model_filepath, "Voting_Classifier") 
#%%
###############################################################################
################################ Test Set scores ##############################
###############################################################################
Test_Set_Results = {}
for clf_name, clf in [("gdb_clf", gdb_clf), ("gnd_clf", gnd_clf),("rbf_svm", rbf_svm), ("final_model", final_model_clf)]:
    clf.fit(X_train, y_train) 
    y_Pred_test      = clf.predict(X_test)
    accuracy_score_  = accuracy_score(y_test,  y_Pred_test)
    precision_score_ = precision_score(y_test, y_Pred_test)
    recall_score_    = recall_score(y_test, y_Pred_test)
    f1_score_        = f1_score(y_test, y_Pred_test)

    Test_Set_Results[clf_name] = {"Accuracy" : accuracy_score_, 
                                  "Precision": precision_score_, 
                                  "Recall"   : recall_score_,
                                  "F1_score" : f1_score_}
    
Test_Set_Results_tab = tabulate(pd.DataFrame(Test_Set_Results).round(3).T , headers="keys", tablefmt='grid')  
with open(rf"{res_filepath}/Test_Set_Results.dat","w") as fileOut:
    fileOut.write(Test_Set_Results_tab)
#%%
###############################################################################
############################### Confusion Matrix ##############################
###############################################################################    
if plotDiagrams == True:
    clf_metricsPlot = ClassificationMetricsPlot(y_test)
    CMatTest = confusion_matrix(y_test, y_Pred_test)
    clf_metricsPlot.Confusion_Matrix_Plot(y_Pred_test, CMatTest, normalize=True,
                                          labels=["Non-Potable","Potable"],
                                          cMap="default",Rotations=[0.,0.],
                                          savePlot=[saveDiagrams,fig_filepath,"Confusion_Matrix"]) 
#%%
###############################################################################
################## Classification Metrics (Precision/Recall) ##################
###############################################################################
if plotDiagrams == True:
    yPred_rbf_svm_test   = rbf_svm.predict(X_test)                                             
    yScores_rbf_svm_test = rbf_svm.decision_function(X_test)
    clf_metricsPlot = ClassificationMetricsPlot(y_test)
    clf_metricsPlot.Precision_Recall_Plots(yPred_rbf_svm_test, yScores_rbf_svm_test, savePlot=[saveDiagrams, fig_filepath, "Precision_Recall"])
    clf_metricsPlot.ROC_Plot(yScores_rbf_svm_test, savePlot=[saveDiagrams,fig_filepath,"ROC"])
# =============================================================================
# #%%
# from sklearn import metrics
# precisions, recalls, thresholds = metrics.precision_recall_curve(y_test, yScores_rbf_svm_test)
# threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]
# y_test_pred_90 = (yScores_rbf_svm_test >= threshold_90_precision)
# =============================================================================