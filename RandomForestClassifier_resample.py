import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, f1_score
from sklearn import tree
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.tree import export_graphviz
import imblearn as imb
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import NearMiss
from joblib import Memory

### Seed
seed = 200  

### Reading .pkl file
num_chr = '1'
pd_df = pd.read_pickle(f"/mnt/wd/nsap/imp2/chr{num_chr}.pkl")
print('Original DF: ', pd_df, sep='\n')

## Saving 1st two columns
classes = pd_df['PHENOTYPE']
pd_df_out = pd_df[['IID','PHENOTYPE']].copy()
pd_df.drop(['IID','PHENOTYPE'], axis=1, inplace=True)

### One-Hot encode
print('Performing one-hot encoding...')
enc = OneHotEncoder(sparse=False, handle_unknown='error')
enc.fit(pd_df)
pd_df_enc = enc.transform(pd_df)
pd_df_columns = enc.get_feature_names_out()
print('Encoded...')

### NearMiss-1 selects the positive samples for which the average distance to the N
### closest samples of the negative class is the smallest.
print('Performing NearMiss undersampling...')
undersample = NearMiss(sampling_strategy=0.5,
                       version=1, n_neighbors=3)
print('Undersampled with NearMiss_1...')

### RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100
                            ,max_features=None
                            ,criterion='gini'
                            ,bootstrap=True
                            ,oob_score=True
                            ,random_state=seed
                            ,class_weight='balanced'
                            ,n_jobs=12)
# clf = clf.fit(training_set, training_classes)
# test_pred_classes = clf.predict(test_set)
# print(Counter(test_pred_classes))

### Define imblearn.pipeline
steps = [('under', undersample), ('model', clf)]
mem = Memory(location='./cachedir')
pipeline = Pipeline(steps=steps, memory=mem)

### Repeated stratified k-fold cross-validation
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=seed)

### Scoring
cv_scores = cross_validate(pipeline
                          ,pd_df_enc
                          ,classes
                          ,scoring=('accuracy'
                                   ,'precision_macro'
                                   ,'precision_micro'
                                   ,'precision_weighted'
                                   ,'recall_macro'
                                   ,'recall_micro'
                                   ,'recall_weighted'
                                   ,'f1_macro'
                                   ,'f1_micro'
                                   ,'f1_weighted')
                          ,cv=cv
                          ,return_estimator=True
                          ,n_jobs=12)
# print(cv_scores)
## Accuracy
print('Accuracy of the model is: ', cv_scores['test_accuracy'])

## Precision
print("Macro precision score of the model: ", cv_scores['test_precision_macro'])
print("Micro precision score of the model: ", cv_scores['test_precision_micro'])
print("Weighted precision score of the model: ", cv_scores['test_precision_weighted'])

## Recall
print("Macro recall score of the model: ", cv_scores['test_recall_macro'])
print("Micro recall score of the model: ", cv_scores['test_recall_micro'])
print("Weighted recall score of the model: ", cv_scores['test_recall_weighted'])

## f1-score
print("Macro F1-score of the model: ", cv_scores['test_f1_macro'])
print("Micro F1-score of the model: ", cv_scores['test_f1_micro'])
print("Weighted F1-score of the model: ", cv_scores['test_f1_weighted'])

# # Grouped Bar Chart for both training and validation data
# def plot_result(x_label, y_label, plot_title, train_data, val_data):
#         '''Function to plot a grouped bar chart showing the training and validation
#           results of the ML model in each fold after applying K-fold cross-validation.
#          Parameters
#          ----------
#          x_label: str, 
#             Name of the algorithm used for training e.g 'Decision Tree'
#           
#          y_label: str, 
#             Name of metric being visualized e.g 'Accuracy'
#          plot_title: str, 
#             This is the title of the plot e.g 'Accuracy Plot'
#          
#          train_result: list, array
#             This is the list containing either training precision, accuracy, or f1 score.
#         
#          val_result: list, array
#             This is the list containing either validation precision, accuracy, or f1 score.
#          Returns
#          -------
#          The function returns a Grouped Barchart showing the training and validation result
#          in each fold.
#         '''
#         
#         # Set size of plot
#         plt.figure(figsize=(12,6))
#         labels = ["1st Fold", "2nd Fold", "3rd Fold", "4th Fold", "5th Fold"]
#         X_axis = np.arange(len(labels))
#         ax = plt.gca()
#         plt.ylim(0.40000, 1)
#         plt.bar(X_axis-0.2, train_data, 0.4, color='blue', label='Training')
#         plt.bar(X_axis+0.2, val_data, 0.4, color='red', label='Validation')
#         plt.title(plot_title, fontsize=30)
#         plt.xticks(X_axis, labels)
#         plt.xlabel(x_label, fontsize=14)
#         plt.ylabel(y_label, fontsize=14)
#         plt.legend()
#         plt.grid(True)
#         plt.show()
