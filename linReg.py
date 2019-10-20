
from sklearn.externals import joblib
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv('timePredictionModel.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values
#for splitting dataset into trining AND testing 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)

#implementing our classifier based 
from sklearn.linear_model import LinearRegression
simplelinearRegression=LinearRegression()
simplelinearRegression.fit(X_train,y_train)
joblib.dump(simplelinearRegression,"timePredictionModel.cpickle")
y_predict=simplelinearRegression.predict(X_test)
y_predict_val=simplelinearRegression.predict(5.5)
 #graph
 
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,simplelinearRegression.predict(X_train))
plt.show()