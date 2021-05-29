import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

def kNN(X_train,X_test,y_train,y_test,n_neighbor):
	knn=KNeighborsClassifier(n_neighbors=n_neighbor)
	knn.fit(X_train,y_train)
	
	trainingaccuracy=knn.score(X_train,y_train)
	testingaccuracy=knn.score(X_test,y_test)
	
	return (trainingaccuracy,testingaccuracy)
	
def randomforest(X_train,X_test,y_train,y_test):
	rf=RandomForestClassifier(n_estimators=100,random_state=0)
	rf.fit(X_train,y_train)
	
	trainingaccuracy=rf.score(X_train,y_train)
	testingaccuracy=rf.score(X_test,y_test)
	
	return (trainingaccuracy,testingaccuracy)
	
def logisticregression(X_train,X_test,y_train,y_test):
	logreg=LogisticRegression(max_iter=1000).fit(X_train,y_train)
	trainingaccuracy=logreg.score(X_train,y_train)
	testingaccuracy=logreg.score(X_test,y_test)
	
	return (trainingaccuracy,testingaccuracy)
	
def decisiontree(X_train,X_test,y_train,y_test):
	tree = DecisionTreeClassifier(random_state=0)
	tree.fit(X_train,y_train)
	trainingaccuracy=tree.score(X_train,y_train)
	testingaccuracy=tree.score(X_test,y_test)
	
	return (trainingaccuracy,testingaccuracy)
def main():
	diabetes=pd.read_csv("diabetes.csv")

	print("Columns of dataset\n",diabetes.columns)

	print("First 5 records of dataset\n",diabetes.head())

	print("Dimension of diabetes data are {}".format(diabetes.shape))


	features=diabetes.iloc[:,:-1]
	labels=diabetes.iloc[:,-1]

	X_train,X_test,y_train,y_test=train_test_split(features,labels,stratify=labels,test_size=0.3,random_state=66)
	
	(trainingaccuracy,testingaccuracy)=kNN(X_train,X_test,y_train,y_test,9)
	print("knn training accuracy is {} and testing accuracy is {}".format(trainingaccuracy,testingaccuracy))
	(trainingaccuracy,testingaccuracy)=randomforest(X_train,X_test,y_train,y_test)
	print("Random Forest training accuracy is {} and testing accuracy is {}".format(trainingaccuracy,testingaccuracy))
	(trainingaccuracy,testingaccuracy)=logisticregression(X_train,X_test,y_train,y_test)
	print("Logistic regression training accuracy is {} and testing accuracy is {}".format(trainingaccuracy,testingaccuracy))
	(trainingaccuracy,testingaccuracy)=decisiontree(X_train,X_test,y_train,y_test)
	print("Decisiontree training accuracy is {} and testing accuracy is {}".format(trainingaccuracy,testingaccuracy))
if __name__=="__main__":
	main()
