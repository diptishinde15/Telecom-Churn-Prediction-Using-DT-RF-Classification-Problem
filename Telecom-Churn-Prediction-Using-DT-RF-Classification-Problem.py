import os
import pandas as pd
import numpy as np

os.chdir("C:/Users/dipti/Documents/IMR/Data_Decision_Tree/")

fullRaw = pd.read_csv('Telecom_Churn.csv')

from sklearn.model_selection import train_test_split
trainDf, testDf = train_test_split(fullRaw, train_size=0.7, random_state = 123)

# Create Source Column in both Train and Test
trainDf['Source'] = 'Train'
testDf['Source'] = 'Test'

# Combine Train and Test
fullRaw = pd.concat([trainDf, testDf], axis = 0)
fullRaw.shape

# Check for NAs
fullRaw.isnull().sum()

# Summarize the data
fullRaw_Summary = fullRaw.describe()

# Remove Customer Id columns
fullRaw.drop(['customerID'], axis = 1, inplace = True)


########################
# Manual recoding of "Dependent Variable"
########################

# We will manually convert our categorical dependent variable to numeric
fullRaw['Churn'] = np.where(fullRaw['Churn'] == 'Yes', 1, 0)

# Class ratio
fullRaw.loc[fullRaw["Source"] == "Train", 'Churn'].value_counts()/fullRaw[fullRaw["Source"] == "Train"].shape[0]



########################
# Dummy variable creation
########################

# Dummy variable creation
fullRaw2 = pd.get_dummies(fullRaw, drop_first = False)
fullRaw2.shape

############################
# Divide the data into Train and Test
############################


# Step 1: Divide into Train and Testest
Train = fullRaw2[fullRaw2['Source_Train'] == 1].drop(['Source_Train', "Source_Test"], axis = 1).copy()
Test = fullRaw2[fullRaw2['Source_Test'] == 1].drop(['Source_Train', "Source_Test"], axis = 1).copy()

########################
# Sampling into X and Y
########################

# Divide each dataset into Indep Vars and Dep var
depVar = 'Churn'
trainX = Train.drop(depVar, axis = 1).copy()
trainY = Train[depVar].copy()
testX = Test.drop(depVar, axis = 1).copy()
testY = Test[depVar].copy()

trainX.shape
testX.shape

########################################
# Decision Tree Model
########################################

from sklearn.tree import DecisionTreeClassifier, plot_tree
from matplotlib.pyplot import figure, savefig, close

M1 = DecisionTreeClassifier(random_state = 123).fit(trainX, trainY) # Indep, Dep


########################################
# Model Visualization
########################################

# Vizualization of DT

figure(figsize = [20, 10]) 
DT_Plot1 = plot_tree(M1, fontsize = 10, feature_names = trainX.columns, 
                     filled = True, class_names = ["0","1"])
############################
# Prediction and Validation on Testset
############################

from sklearn.metrics import classification_report

# Prediction on Testset
Test_Pred = M1.predict(testX)

# Classification Model Validation
Confusion_Mat = pd.crosstab(testY, Test_Pred)
Confusion_Mat 

# Validation on Testset
print(classification_report(testY, Test_Pred)) 


############
# DT Model 2
############

# Build Model
M2 = DecisionTreeClassifier(random_state = 123, min_samples_leaf = 500).fit(trainX, trainY)

# Vizualization of DT
figure(figsize = [20, 10])
DT_Plot2 = plot_tree(M2, fontsize = 10, feature_names = trainX.columns, 
                     filled = True, class_names = ["0","1"])

# Prediction on Testset
Test_Pred = M2.predict(testX)

# Classification Model Validation
Confusion_Mat = pd.crosstab(testY, Test_Pred)
Confusion_Mat # R, C format

# Validation on Testset
print(classification_report(testY, Test_Pred)) # Actual, Predicted


########################################
# Random Forest
########################################

from sklearn.ensemble import RandomForestClassifier

M1_RF = RandomForestClassifier(random_state = 123).fit(trainX, trainY)

# Prediction on Testset
Test_Pred = M1_RF.predict(testX)

# Confusion Matrix
Confusion_Mat = pd.crosstab(testY, Test_Pred) # R, C format
Confusion_Mat 

# Validation on Testset
print(classification_report(testY, Test_Pred)) # Actual, Predicted


# Variable importance
M1_RF.feature_importances_

Var_Importance_Df = pd.concat([pd.DataFrame(M1_RF.feature_importances_),
                               pd.DataFrame(trainX.columns)], axis = 1)

Var_Importance_Df
Var_Importance_Df.columns = ["Value", "Variable_Name"]
Var_Importance_Df.sort_values("Value", ascending = False, inplace = True)
Var_Importance_Df

import seaborn as sns
plot = sns.scatterplot(x = "Variable_Name", y = "Value", data = Var_Importance_Df) 


#################
# RF Model with tuning parameters/ hyperparameter tuning
#################

M2_RF = RandomForestClassifier(random_state = 123, n_estimators = 25, 
                               max_features = 5, min_samples_leaf = 500)
M2_RF = M2_RF.fit(trainX, trainY)
Test_Pred = M2_RF.predict(testX)

# Confusion Matrix
Confusion_Mat = pd.crosstab(testY, Test_Pred) 
Confusion_Mat 

# Validation on Testset
print(classification_report(testY, Test_Pred)) 

#################
# Manual Grid Searching
#################

n_estimators_List = [25, 50, 75] # range(25,100,25)
max_features_List = [5, 7, 9] # range(5,11,2)
min_samples_leaf_List = [5, 10, 25, 50]

Counter = 0

Tree_List = []
Num_Features_List = []
Samples_List = []
Accuracy_List = []

Model_Validation_Df = pd.DataFrame()
Model_Validation_Df2 = pd.DataFrame()
Model_Validation_Df3 = pd.DataFrame()

for i in n_estimators_List:    
    for j in max_features_List:        
        for k in min_samples_leaf_List:                        
            
            Counter = Counter + 1
            print(Counter)
#            print(i,j,k)            
            Temp_Model = RandomForestClassifier(random_state=123, n_estimators = i, 
                                                max_features = j, min_samples_leaf = k).fit(trainX, trainY)
            Test_Pred = Temp_Model.predict(testX)                 
            Confusion_Mat = pd.crosstab(testY, Test_Pred)
            Temp_Accuracy = (sum(np.diag(Confusion_Mat))/testY.shape[0])*100            
            print(i,i,k,Temp_Accuracy)
            
            # Alternate 1
            Tree_List.append(i)
            Num_Features_List.append(j)
            Samples_List.append(k)
            Accuracy_List.append(Temp_Accuracy)
            
            # Alternate 2
            tempDf = pd.DataFrame([[i,j,k,Temp_Accuracy]]) 
            Model_Validation_Df2 = Model_Validation_Df2.append(tempDf)
            
            # Alternate 3
            tempDf = pd.DataFrame([[i,j,k,Temp_Accuracy]])
            Model_Validation_Df3 = pd.concat([Model_Validation_Df3, tempDf], axis = 0)
            
            
Model_Validation_Df = pd.DataFrame({'Trees': Tree_List, 'Max_Features': Num_Features_List, 
                                    'Min_Samples_Leaf': Samples_List, 'Accuracy': Accuracy_List})
    
Model_Validation_Df2.columns = ['Trees', 'Max_Features', 'Min_Samples_Leaf', 'Accuracy']
Model_Validation_Df3.columns = ['Trees', 'Max_Features', 'Min_Samples_Leaf', 'Accuracy']


########################################
# Random Forest using GridSearchCV
########################################

from sklearn.model_selection import GridSearchCV

n_estimators_List = [25, 50, 75] # range(25,100,25)
max_features_List = [5, 7, 9] # range(5,11,2)
min_samples_leaf_List = [5, 10, 25, 50]

my_param_grid = {'n_estimators': n_estimators_List, 
                 'max_features': max_features_List, 
                 'min_samples_leaf' : min_samples_leaf_List} 

# 3-Fold CV
Grid_Search_Model = GridSearchCV(estimator = RandomForestClassifier(random_state=123), 
                     param_grid=my_param_grid,  
                     scoring='accuracy', 
                     cv=3).fit(trainX, trainY) 


Model_Validation_Df4 = pd.DataFrame.from_dict(Grid_Search_Model.cv_results_)

# Grid_Search_Model.cv_results_

# 5-Fold CV
Grid_Search_Model = GridSearchCV(estimator = RandomForestClassifier(random_state=123), 
                     param_grid=my_param_grid,  
                     scoring='accuracy', 
                     cv=5).fit(trainX, trainY)
Model_Validation_Df5 = pd.DataFrame.from_dict(Grid_Search_Model.cv_results_)


# 5-Fold CV with parallel processing
Grid_Search_Model = GridSearchCV(estimator = RandomForestClassifier(random_state=123), 
                     param_grid=my_param_grid,  
                     scoring='accuracy', 
                     cv=5, n_jobs = -1).fit(trainX, trainY) 
Model_Validation_Df6 = pd.DataFrame.from_dict(Grid_Search_Model.cv_results_)


######
# Final Model on entire trainset using the best parameters found through grid search
######

RF_Final = RandomForestClassifier(random_state = 123, n_estimators = 50, 
                               max_features = 9, min_samples_leaf = 5).fit(trainX, trainY)
Test_Pred = RF_Final.predict(testX)

# Confusion Matrix
Confusion_Mat = pd.crosstab(testY, Test_Pred) # R, C format (Actual = testY, Predicted = Test_Pred)
Confusion_Mat 

# Validation on Testset
print(classification_report(testY, Test_Pred)) # Actual, Predicted
