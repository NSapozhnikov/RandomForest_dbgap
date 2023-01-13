source("pca_helpers.R")

pref = '/mnt/wd/nsap/Classifiers/RandomForest_sklearn/data.clean.pruned' 

data <- LoadData2(pref)

PlotPCA(data, output = sprintf("%s.pca.png", pref))
