import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Reading the dataset
df = pd.read_csv('Dataset.csv', encoding='latin1') #Ensure that the file path is correct

X = df.drop(columns='Insurance')
y = df.iloc[:, 3]
print(df.head())

#Normalizing the values
for i in range(len(X.columns)):
  X.iloc[:, i] = X.iloc[:, i]/np.max(X.iloc[:, i])
print(X)

y = y/np.max(y)

def hypothesis(thetas, X):
  y_pred = []
  for i in range(len(X_1)):
    y_pred.append(thetas[0] + thetas[1]*X[0][i] + thetas[2]*X[1][i] + thetas[3]*X[2][i])
  return y_pred

X_1 = X.iloc[:, 0].values.tolist()
X_2 = X.iloc[:, 1].values.tolist()
X_3 = X.iloc[:, 2].values.tolist()

X = [X_1,X_2,X_3]

thetas = [1,0.1,0.1,0.1]

y_pred = hypothesis(thetas,X)

def cost_fn(y_pred, y):
  sq_error = 0
  J = 0
  for i in range(len(y_pred)):
    sq_error += (y_pred[i] - y[i])**2
  J = sq_error/len(y_pred)
  return J

cost_fn(y_pred, y)

def gradient_descent(thetas, X, y, alpha, epoch):
  N = len(y)
  J = []
  y_pred_list = []
  for i in range(epoch):
    y_pred = hypothesis(thetas, X)
    #computing new val for thetas
    sum_0,sum_1,sum_2,sum_3 = 0,0,0,0

    for j in range(N):
      sum_0 += (y_pred[j] - y[j])
      sum_1 += (y_pred[j] - y[j])*X[0][j]
      sum_2 += (y_pred[j] - y[j])*X[1][j]
      sum_3 += (y_pred[j] - y[j])*X[2][j]

    thetas[0] -= (2/N)*sum_0*alpha
    thetas[1] -= (2/N)*sum_1*alpha
    thetas[2] -= (2/N)*sum_2*alpha
    thetas[3] -= (2/N)*sum_3*alpha
    J.append(cost_fn(y_pred, y))
    y_pred_list.append(y_pred)
  return J,y_pred_list, thetas

epochs = 1500
alpha = 0.5
values = gradient_descent(thetas, X, y, alpha, epochs)

loss = values[0]
min_loss = min(loss)
min_loss_val = loss.index(min_loss)
final_coeffs = values[2]
print(min_loss_val,min_loss, final_coeffs)

y_pred_list = values[1]
print(y_pred_list[min_loss_val])

plt.figure()
plt.scatter(x=list(range(0, len(y_pred))),y= y_pred_list[min_loss_val], color='blue')         
plt.scatter(x=list(range(0, len(y_pred))), y=y, color='red')
plt.show()

plt.figure()
plt.scatter(x=list(range(0, epochs)),y= loss, color='blue')      
plt.show() 