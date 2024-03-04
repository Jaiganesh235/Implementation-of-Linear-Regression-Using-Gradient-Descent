# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the needed packages
2. Read the txt file using read_csv
3. Use numpy to find theta,x,y values
4. To visualize the data use plt.plot

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: S.JAIGANESH
RegisterNumber: 212222240037 
*/
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.1,num_iters=1000):
X=np.c_[np.ones(len(X1)),X1]
theta=np.zeros(X.shape[1]).reshape(-1,1)
for _ in range(num_iters):
predictions=(X).dot(theta).reshape(-1,1)
errors=(predictions-y).reshape(-1,1)
theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
return theta

data=pd.read_csv("/content/50_Startups.csv",header=None)
data.head()

X=(data.iloc[1:,:-2].values)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_scaled=scaler.fit_transform(X1)
y1_scaled=scaler.fit_transform(y)
print(X)
print(X1_scaled)

theta=linear_regression(X1_scaled,y1_scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_scaled),theta)
prediction=prediction .reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")

```

## Output:
![Screenshot 2024-03-04 143539](https://github.com/Jaiganesh235/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118657189/07e71cd9-2cd8-4d1b-9212-ebe39fe4c4ce)
![Screenshot 2024-03-04 144018](https://github.com/Jaiganesh235/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118657189/9503808e-0dd0-444b-b573-5901d0e2ae8c)
![Screenshot 2024-03-04 144039](https://github.com/Jaiganesh235/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118657189/0d46d6ad-3380-48a2-9dea-94118f9f2d44)
![Screenshot 2024-03-04 194734](https://github.com/Jaiganesh235/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118657189/a0a17742-1dd1-48f2-ab32-262cb90fb115)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
