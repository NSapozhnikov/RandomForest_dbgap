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

### so that the result is reproducible
seed = 100  

# for num_chr in range(1,23):
num_chr = '1'
pd_df = pd.read_pickle(f"/mnt/wd/nsap/imp2/chr{num_chr}.pkl")
print('Original DF: ', pd_df, sep='\n')
### Forming list of general classes
general_classes = pd_df['PHENOTYPE']
print('Amount of original labels: ', Counter(general_classes), sep='\n')

### Drop IID column -> general_set
print('Droping IID and PHENOTYPE columns...')
pd_df.drop(columns=['IID'], inplace=True)
pd_df.drop(columns=['PHENOTYPE'], inplace=True)
                                                        
### One-Hot encode
print('Performing one-hot encoding...')
enc = OneHotEncoder(sparse=False, handle_unknown='error')
enc.fit(pd_df)
pd_df_enc = enc.transform(pd_df)
pd_df_columns = enc.get_feature_names_out()
print('Encoded...')
# print('Encoded DF: ', pd_df_enc, sep='\n')
# print('pd_df_columns: ', pd_df_columns, sep='\n')

### Split training set from general_set
print('Splitting dataset...')
training_set, test_set, training_classes, test_classes = train_test_split(pd_df_enc
                                                        ,general_classes
                                                        ,test_size=0.333
                                                        ,random_state = seed)
# print('training_set: ', training_set, sep='\n')
# print('test_set: ', test_set, sep='\n')
# print('training_classes: ', training_classes, sep='\n')
# print('test_classes: ', test_classes, sep='\n')

# ### Forming list of features
# feature_names = test_set.columns
# # print('feature_names: ', list(feature_names), sep='\n')

### RandomForestClassifier
# Building forest
print('Building forest...')
clf = RandomForestClassifier(n_estimators=100
                            ,max_features=None
                            ,criterion='gini'
                            ,bootstrap=True
                            ,oob_score=True
                            ,random_state=seed
                            ,class_weight='balanced_subsample'
                            ,n_jobs=-1)
clf = clf.fit(training_set, training_classes)
test_pred_classes = clf.predict(test_set)
print(Counter(test_pred_classes))

### Scoring the model
## Accuracy
acc_score = accuracy_score(test_classes, test_pred_classes)
print('Accuracy of the model is: ', acc_score)

## f1-score
print("Macro F1-score of the model: ", f1_score(test_classes, test_pred_classes
                                                ,average='macro'))
print("Micro F1-score of the model: ", f1_score(test_classes, test_pred_classes
                                                ,average='micro'))
print("Weighted F1-score of the model: ", f1_score(test_classes, test_pred_classes
                                                ,average='weighted'))

## Precision
print("Macro precision score of the model: ", precision_score(test_classes, test_pred_classes
                                                              ,average='macro'))
print("Micro precision score of the model: ", precision_score(test_classes, test_pred_classes
                                                              ,average='micro'))
print("Weighted precision score of the model: ", precision_score(test_classes, test_pred_classes
                                                              ,average='weighted'))

## Recall
print("Macro recall score of the model: ", recall_score(test_classes, test_pred_classes
                                                        ,average='macro'))
print("Micro recall score of the model: ", recall_score(test_classes, test_pred_classes
                                                        ,average='micro'))
print("Weighted recall score of the model: ", recall_score(test_classes, test_pred_classes
                                                        ,average='weighted'))



# 
# # ## ROC_AUC
# # train_set_probs = clf.predict_proba(training_set)[:,2] 
# # test_set_probs = clf.predict_proba(test_set)[:, 1]
# # train_predictions = clf.predict(training_set)
# # print('Train ROC AUC Score: ', roc_auc_score(training_classes, train_set_probs))
# # print('Test ROC AUC  Score: ', roc_auc_score(test_classes, test_set_probs))
# #
# #
# ### Scoring feature importances
# feature_imp = pd.Series(clf.feature_importances_,
#                         index = feature_names).sort_values(ascending = False)
# print(feature_imp[0:10])
# ### Plot one tree
# ## Graphviz
# export_graphviz(clf.estimators_[5], 
#                 out_file='rf_individualtree.dot', 
#                 feature_names=feature_names,
#                 class_names=general_classes,
#                 rounded = True, proportion = False, 
#                 precision = 2, filled = True)

# ## Matplotlib
# fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (10,10), dpi=1000)
# tree.plot_tree(clf.estimators_[50],
#                feature_names = feature_names,
#                class_names=general_classes,
#                filled = True,
#                rounded=True);
# fig.savefig('rf_individualtree.png')

