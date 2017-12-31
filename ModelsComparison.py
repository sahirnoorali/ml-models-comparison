# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 17:54:05 2017

Author: Sahir
Code:   Apply Models on 10 Data-Sets. Calculate accuracy, fmeasure and auc 
        for each. Statistical analysis using WIN-TIE-LOSE and T-test
"""
#-------------------------------------------------------------------------
# All the Libraries 
#------------------------------------------------------------------------- 
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import  roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.cross_validation import KFold
from sklearn.metrics import f1_score
from scipy.stats import ttest_ind
import warnings
import csv
warnings.filterwarnings("ignore")
#-------------------------------------------------------------------------
# Declare the Total Global Storage of Performance Measures 
#-------------------------------------------------------------------------  
std_Accuracy = [[0 for y in range(0,9)] for x in range(0,10)]
Accuracy = [[0 for y in range(0,9)] for x in range(0,10)]
std_fmeasure = [[0 for y in range(0,9)] for x in range(0,10)] 
Fmeasure = [[0 for y in range(0,9)] for x in range(0,10)]    
std_AUC = [[0 for y in range(0,9)] for x in range(0,10)] 
AUC = [[0 for y in range(0,9)] for x in range(0,10)]                               
#------------------------------------------------------------------------
# Function to Apply Stacking (KMeans then KNN)
#------------------------------------------------------------------------
def Stacking(model, X, Y, folds, modelName):     

    #Predict clusters with KMeans
    pred = model.fit_predict(X)
	
    #Append to the dataset
    X = np.c_[X, pred]
	
    #Init KNN
    knn = KNeighborsClassifier(n_neighbors=3)
	
    #Init arrays for performance measures
    acc_array = [0] * 10
    f1_array =  [0] * 10
    auc_array = [0] * 10

    #10 times K Fold
    for i in range (1,11):
        kf = KFold(X.shape[0],n_folds=folds, random_state=i)
        acc = 0
        f1  = 0
        auc_ = 0
        for train_index, test_index in kf:
            #Get the train and test (X & Y)
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]
            #Train the Model
            knn.fit(X_train, Y_train) 
            #Predict
            prediction = knn.predict(X_test)
            #Get Accuracy    
            acc = acc + accuracy_score(Y_test,prediction)
            #Get AUC
            if all(item == 1 for item in Y_test) == True:
                auc_ = auc_ + 1
            elif all(item == 0 for item in Y_test) == True:
                auc_ = auc_ + 0
            else:
                auc_ = auc_ + roc_auc_score(Y_test, prediction)
            #Get F1
            f1 = f1 + f1_score(Y_test,prediction)
        
	#Store the performance measures (10 values for each)
        acc_array[i-1] = acc/folds
        f1_array[i-1]  = f1/folds
        auc_array[i-1] = auc_/folds

    #Return the performance measures
    return acc_array,f1_array,auc_array
#------------------------------------------------------------------------
# General Function to Apply Models and Return Performance Measures
#------------------------------------------------------------------------
def ApplyModel(model, kf, X, Y, folds, modelName):    
    i = 1
    acc = 0.0
    f1 = 0.0
    auc_ = 0.0    
    #Loop Fold times
    for train_index, test_index in kf:
        #Get the train and test (X & Y)
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        #Train the Model
        model.fit(X_train, Y_train) 
        #Predict
        pred = model.predict(X_test)
        #Get Accuracy    
        acc = acc + accuracy_score(Y_test,pred)
        #Get AUC
        if all(item == 1 for item in Y_test) == True:
            auc_ = auc_ + 1
        elif all(item == 0 for item in Y_test) == True:
            auc_ = auc_ + 0
        else:
            auc_ = auc_ + roc_auc_score(Y_test, pred)
        #Get F1
        f1 = f1 + f1_score(Y_test,pred)
        i = i+1
    return acc/folds,f1/folds,auc_/folds;
#-------------------------------------------------------------------------
# Helper Function To Pass Models And Store Results 
#------------------------------------------------------------------------- 
def Compute(model, X, Y, shape, folds, modelName, row, col):
    #Init arrays for performance measures
    acc_array = [0] * 10
    f1_array  = [0] * 10 
    auc_array = [0] * 10

    #Apply Stacking if passed
    if modelName == "Stacking":
        acc_array , f1_array, auc_array = Stacking(model, X, Y, folds, modelName)
    
    else:
	#10 times K fold
        for i in range (1,11):  
        #Defining a fold of 10
            folds = 10
            kf = KFold(shape,n_folds=folds, random_state=i)
	    #Apply the passed model
            acc_array[i-1] , f1_array[i-1], auc_array[i-1] =  ApplyModel(model, kf, X, Y, folds, modelName)

    #Store all the pefromance measures in the global storage
    Accuracy[row][col]     = round(np.array(acc_array).mean(), 4)        
    Fmeasure[row][col]     = round(np.array(f1_array).mean(), 4)
    AUC[row][col]          = round(np.array(auc_array).mean(), 4)      
    std_Accuracy[row][col] = round(np.std(acc_array),4)
    std_fmeasure[row][col] = round(np.std(f1_array),4) 
    std_AUC[row][col]      = round(np.std(auc_array),4) 
     
    print("Mean Accuracy of 10 x 10 Cross Validation of",modelName," : ",np.array(acc_array).mean())
    print("F-Measure of 10 x 10 Cross Validation of",modelName,": ",np.array(f1_array).mean())
    print("AUC of 10 x 10 Cross Validation of",modelName,": ",np.array(auc_array).mean(),"\n")
#-------------------------------------------------------------------------
# Setup According To The Datasets
#-------------------------------------------------------------------------  
def DataSet(dataset,dName):
    
    if dName == "Abalone":
        d = {'M' : 0, 'F' : 1, "I" : 2}
        dataset[0] = dataset[0].map(d)
        dataset[8] = np.where(dataset[8] >= 20, 0, 1)
        array = dataset.values
        X = array[:,0:7]
        Y = array[:,8]

    if dName == "Balance Scale":
        dataset[0] = np.where(dataset[0] == 'B', 0, 1)
        array = dataset.values
        X = array[:,1:4]
        Y = array[:,0]

    if dName == "CMC":
        dataset[9] = np.where(dataset[9] == 2, 0, 1)
        array = dataset.values
        X = array[:,0:8]
        Y = array[:,9]

    if dName == "Glass":
        dataset[10] = np.where(dataset[10] == 3, 0, 1)
        array = dataset.values
		#Did not take ID
        X = array[:,1:9] 
        Y = array[:,10]
    
    if dName == "Housing":
        dataset[13] = np.where(np.logical_and(dataset[13] >= 20, dataset[13] <= 23 ), 0, 1)
        array = dataset.values
        X = array[:,0:12]
        Y = array[:,13]

    if dName == "Haberman":
        dataset[3] = np.where(dataset[3] == 2, 0, 1)
        array = dataset.values
        X = array[:,0:2] 
        Y = array[:,3]

    if dName == "HSLog":
        dataset[13] = np.where(dataset[13] == 2, 0, 1)
        array = dataset.values
        X = array[:,0:12] 
        Y = array[:,13]

    if dName == "Ionosphere":
        dataset[34] = np.where(dataset[34] == 'b', 0, 1)
        array = dataset.values
        X = array[:,0:33] 
        Y = array[:,34]

    if dName == "Nursery":
        d = {'usual': 0, 'pretentious': 1, 'great_pret': 2}
        dataset[0] = dataset[0].map(d)
        d = {'proper': 0, 'less_proper': 1, 'improper': 2, 'critical':3, 'very_crit': 4}
        dataset[1] = dataset[1].map(d)
        d = {'complete' : 0, 'completed': 1, 'incomplete' : 2, 'foster' : 3}
        dataset[2] = dataset[2].map(d)
        s = {'more' : 3, '1' : 0, '2' : 1, '3' : 2}
        dataset[3] = dataset[3].map(s)
        d = {'convenient' : 0, 'less_conv' : 1, 'critical' : 2}
        dataset[4] = dataset[4].map(d)
        dataset[5] = np.where(dataset[5] == 'convenient',0 , 1)
        d = {'nonprob' : 0, 'slightly_prob' : 1, 'problematic' : 2}
        dataset[6] = dataset[6].map(d)
        d = {'recommended' : 0, 'priority' : 1, 'not_recom' : 2}
        dataset[7] = dataset[7].map(d)
        dataset[8] = np.where(dataset[8] == 'very_recom', 0, 1)
        array = dataset.values
        X = array[:,0:7] 
        Y = array[:,8]
        
    if dName == "Phenome":
        array = dataset.values
        X = array[:,0:4] 
        Y = array[:,5]
        
	return X,Y,array.shape[0]
#-------------------------------------------------------------------------
# Fill the Win Tie Lose Table 
#------------------------------------------------------------------------- 
def FillWTL(WTL_Measure, MeasureArray):
    for i in range(0,8):
        j = 0
        z = i+1
        while (z < 9):
            WTL = [0,0,0]
            j=0
            while (j <= 9):
                if round(MeasureArray[j][i],3) > round(MeasureArray[j][z],3):
                    WTL[0]+=1
                elif round(MeasureArray[j][i],3) == round(MeasureArray[j][z],3):
                    WTL[1]+=1
                else: 
                    WTL[2]+=1
                j+=1
            WTL_Measure[i][z] = WTL
            z+=1
    return WTL_Measure
#-------------------------------------------------------------------------
# Fill the T test Tables 
#------------------------------------------------------------------------- 
def FillT(T_Values,P_Values, MeasureArray):
    for i in range(0,8):
        j = 0
        z = i+1
        while (z < 9):
            a = []
            b = []
            j=0
            while (j <= 9):
                a.append(MeasureArray[j][i])
                b.append(MeasureArray[j][z])
                j+=1
            T_Values[i][z],P_Values[i][z] = ttest_ind(a, b)
            z+=1
    return T_Values,P_Values
#-------------------------------------------------------------------------
# Output Performance Measure: 
#------------------------------------------------------------------------- 
def OutputMeasure(Measure, std_Measure, dataset_names, writer):
    writer.writerow("\n")
    i = 0
    for v1,v2 in zip(Measure,std_Measure):
        row = [dataset_names[i]]
        for j in range(0,9):
            row.append(v1[j])
        writer.writerow(row)
        
        row = [" "]
        for j in range(0,9):
            row.append(v2[j])
        writer.writerow(row)
    
        writer.writerow("\n")
        i+=1
    writer.writerow("\n")     
#-------------------------------------------------------------------------
# Output WIN-TIE-LOSE: 
#-------------------------------------------------------------------------     
def OutputWTL(WTL,P,T,model_names,writer):
    for i in range(0,9):
        writer.writerow([model_names[i]])
        row = ["s"]
        for j in range(0,9):
            row.append(WTL[i][j])
        writer.writerow(row)
        row = ["p"]
        for j in range(0,9):
            row.append(P[i][j])
        writer.writerow(row)
        row = ["t"]
        for j in range(0,9):
            row.append(T[i][j])
        writer.writerow(row)
        
    writer.writerow("\n") 
#-------------------------------------------------------------------------
# Main Program Flow: 
#------------------------------------------------------------------------- 
models = [BaggingClassifier(), RandomForestClassifier(), AdaBoostClassifier(), 
          KNeighborsClassifier(n_neighbors=3), SVC(kernel='linear', C=1), SVC(kernel='rbf', C=1), 
          GaussianNB(),DecisionTreeClassifier(), KMeans(n_clusters=5)]

model_names = ["Bagging with DT", "Random Forest", "AdaBoost", "3NN", "Linear SVM", "RBF SVM",
         "Naive Bayes","Decision Tree","Stacking"]

#----------------------------------------------------------------
#Read the Data Sets
#----------------------------------------------------------------
abalone = pd.read_csv("datasets/abalone/abalone.data", header=None)
balance_scale = pd.read_csv("datasets/balance-scale/balance-scale.data",header=None)
cmc = pd.read_csv("datasets/CMC/cmc.data",header=None)
glass = pd.read_csv("datasets/Glass/glass.data",header=None)
housing = pd.read_table('datasets/housing/housing.data',  sep='\s+', header=None)   
haberman = pd.read_csv("datasets/haberman/haberman.data",header=None)
hslog = pd.read_table("datasets/Heart-statlog/heart.dat",sep=' ', header=None)
ionosphere = pd.read_csv("datasets/Ionosphere/ionosphere.data",header=None)
nursery = pd.read_csv("datasets/nursery/nursery.data",header=None)
phenome = pd.read_csv("datasets/phoneme/phoneme.dat",header=None)

datasets = [abalone, balance_scale, cmc, glass, housing, haberman, hslog, ionosphere, nursery, phenome]
dataset_names = ["Abalone","Balance Scale","CMC","Glass","Housing", "Haberman","HSLog", "Ionosphere","Nursery", "Phenome"]

folds = 10
#----------------------------------------------------------------
#Main computation loop
#----------------------------------------------------------------
for i in range(0,10):
    print(dataset_names[i],"\n")
    X,Y,shape = DataSet(datasets[i],dataset_names[i])
    j=0
    for model, name in zip(models,model_names):
        if j == 9 :
            j = 0
        Compute(model,X,Y,shape,folds,name,i,j)
        j+=1
#----------------------------------------------------------------
#Fill in the Win-Tie-Lose Tables     
#----------------------------------------------------------------
WTL_Acc = [[[0 for k in range(3)] for j in range(9)] for i in range(9)]
WTL_Acc = FillWTL(WTL_Acc, Accuracy)

WTL_Auc = [[[0 for k in range(3)] for j in range(9)] for i in range(9)]
WTL_Auc = FillWTL(WTL_Auc, AUC)

WTL_Fmeasure = [[[0 for k in range(3)] for j in range(9)] for i in range(9)]
WTL_Fmeasure = FillWTL(WTL_Fmeasure, Fmeasure)
#----------------------------------------------------------------
#Fill in the T-Test Tables     
#----------------------------------------------------------------
T_Values_Acc = [[0 for j in range(9)] for i in range(9)]
P_Values_Acc = [[0 for j in range(9)] for i in range(9)]
T_Value_Acc, P_Values_Acc = FillT(T_Values_Acc,P_Values_Acc, Accuracy)

T_Values_Auc = [[0 for j in range(9)] for i in range(9)]
P_Values_Auc = [[0 for j in range(9)] for i in range(9)]
T_Value_Auc, P_Values_Auc = FillT(T_Values_Auc,P_Values_Auc, AUC)

T_Values_Fmeasure = [[0 for j in range(9)] for i in range(9)]
P_Values_Fmeasure = [[0 for j in range(9)] for i in range(9)]
T_Value_Fmeasure, P_Values_Fmeasure = FillT(T_Values_Fmeasure,P_Values_Fmeasure, Fmeasure)
#----------------------------------------------------------------
# Output Results to a csv file
#----------------------------------------------------------------    
fl = open('Results.csv', 'w',newline="\n", encoding="utf-8")

writer = csv.writer(fl)

m_str = [" "]
for i in range(len(model_names)):
    m_str.append(model_names[i])
#---------------------------------------------------------------- 
#Output all the Accuracy   
#---------------------------------------------------------------- 
writer.writerow(["","","","","Accuracy",""]) 
writer.writerow(m_str)
    
OutputMeasure(Accuracy,std_Accuracy,dataset_names,writer)    
#---------------------------------------------------------------- 
#Output all the F1 Measure
#---------------------------------------------------------------- 
writer.writerow(["","","","","F1-Measure",""])
writer.writerow(m_str)

OutputMeasure(Fmeasure,std_fmeasure,dataset_names,writer)    
#---------------------------------------------------------------- 
#Output all the AUC
#---------------------------------------------------------------- 
writer.writerow(["","","","","AUC",""])
writer.writerow(m_str)

OutputMeasure(AUC,std_AUC,dataset_names,writer)
#---------------------------------------------------------------- 
#Output the WIN-TIE-LOSE for Accuracy
#---------------------------------------------------------------- 
writer.writerow(["","","","WIN, TIE, LOSE","-","Accuracy"])   
writer.writerow(m_str)

OutputWTL(WTL_Acc,P_Values_Acc,T_Value_Acc,model_names,writer)
#---------------------------------------------------------------- 
#Output the WIN-TIE-LOSE for F-Measure
#---------------------------------------------------------------- 
writer.writerow(["","","","WIN, TIE, LOSE","-","Fmeasure"])  
writer.writerow(m_str) 

OutputWTL(WTL_Fmeasure,P_Values_Fmeasure,T_Value_Fmeasure,model_names,writer)
#---------------------------------------------------------------- 
#Output the WIN-TIE-LOSE for AUC
#---------------------------------------------------------------- 
writer.writerow(["","","","WIN, TIE, LOSE","-","AUC"])
writer.writerow(m_str)
    
OutputWTL(WTL_Auc,P_Values_Auc,T_Value_Auc,model_names,writer)

fl.close() 
#----------------------------------------------------------------

