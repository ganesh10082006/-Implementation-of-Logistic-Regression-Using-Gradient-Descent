# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages required.
2. Read the dataset.
3. Define X and Y array.
4. Define a function for sigmoid, loss, gradient and predict and perform operations.
## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: GANESH G. 
RegisterNumber:  212223230059
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv("Placement_Data.csv")

dataset

dataset=dataset.drop("sl_no",axis=1)
dataset=dataset.drop("salary",axis=1)

dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')

dataset.dtypes

dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes

dataset

X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:, -1].values

Y

theta=np.random.randn(X.shape[1])

y=Y

def sigmoid(z):
    return 1/(1+np.exp(-z))

def loss(theta,X,y):
    h=sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h) + (1-y) * np.log(1-h))

def gradient_descent(theta,X,y,alpha,num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y)/m
        theta -= alpha*gradient
    return theta
    
theta = gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)

def predict(theta,X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h>=0.5,1,0)
    return y_pred
    
y_pred = predict(theta,X)

accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy:", accuracy)

print(Y)

xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)

print(y_prednew)

xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)

print(y_prednew)

```

## Output:

### 1. Read the file and display:
![image](https://github.com/user-attachments/assets/e3513cf0-7728-48e9-9937-7b95125c1f22)

### 2. Categorizing columns:
![image](https://github.com/user-attachments/assets/1fc21715-b2d0-4d5b-b515-6735298fb986)

### 3. Labelling columns and displaying dataset:
![image](https://github.com/user-attachments/assets/2eabefc8-6ea3-497d-a4dc-f35641b87876)

### 4. Display dependent variable:
![image](https://github.com/user-attachments/assets/690b8053-fa04-421a-85aa-2cd507d8b5a0)

### 5.Printing accuracy:
![image](https://github.com/user-attachments/assets/231e91dc-e61e-4731-8ed6-93e34ae5ba04)

### 6. Printing Y:
![image](https://github.com/user-attachments/assets/696751bc-4ff2-4fba-82f0-e98ff6afee88)

### 7. Printing y_prednew:
![image](https://github.com/user-attachments/assets/79956b0e-463e-4a3b-ba0c-b8accf6c5fa9)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

