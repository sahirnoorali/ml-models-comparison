# Analysis and Comparison of Machine Learning Models

This project compares and evaluates various Machine Learning Algorithms using F-measure, Accuracy and AUC (Area Under Curve).
The Machine Learning Models applied are as follows:

1. Bagging with Decision Tree
2. Random Forest
3. AdaBoost
4. 3-NN
5. SVM with Linear Kernel
6. SVM with RBF Kernel
7. Naive Bayes
8. Decision Tree
9. Kmeans (5 clusters) with 3-NN classifier -> Stacking

The datasets used to compare the above models are listed below:

1. Abalone (https://archive.ics.uci.edu/ml/datasets/abalone)
2. Balance Scale (http://archive.ics.uci.edu/ml/datasets/balance+scale)
3. CMC (https://archive.ics.uci.edu/ml/datasets/Contraceptive+Method+Choice)
4. Glass (https://archive.ics.uci.edu/ml/datasets/glass+identification)
5. Housing (https://archive.ics.uci.edu/ml/machine-learning-databases/housing/)
6. Haberman (https://archive.ics.uci.edu/ml/datasets/Haberman%27s+Survival)
7. HSLog (http://archive.ics.uci.edu/ml/datasets/statlog+(heart))
8. Ionosphere (https://archive.ics.uci.edu/ml/datasets/ionosphere)
9. Nursery (https://archive.ics.uci.edu/ml/datasets/nursery)
10. Phenome (uploaded)

Each model is applied on each dataset with a 10x10 Fold Cross Validation and a comprehensive table for each performance measure
(F-measure, Accuracy and AUC) is written to 'Results.csv'. Statistical analysis via t-test and WIN-TIE-LOSS is also performed and 
a table for each performance measure is again written to the same CSV file. These 6 tables are formatted properly and explained in
the document 'Report.docx' uploaded.

Note: Each table compares a specific performance measure of the given models with respect to each dataset.
