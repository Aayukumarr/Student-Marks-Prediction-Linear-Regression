# Train and Test the model 
# from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
x=np.array([10,20,30,40,50]).reshape(-1,1)
y=np.array([1,2,3,4,5])
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("Predicted values:",y_pred)
print("x_train:",x_train)
print("x_test:",x_test)
print("y_train:",y_train)
print("y_test:",y_test)

# accuracy check of model
#  Next Step: Model Kitna Sahi Hai? (Accuracy Check)

# Abhi model ne predict to kar diya…
# But ? Kya prediction sahi hai ya galat?

# Isko check karne ke liye use hota hai:

#  Error Calculation
# 1. MAE → Mean Absolute Error
# 2. MSE → Mean Squared Error
# 3. RMSE → Root Mean Squared Error ⭐ Important
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
print("MSE:",mse)
print("MAE:",mae)
print("RMSE:",rmse)
# now making the graph with matplotlib
import matplotlib.pyplot as plt
plt.scatter(x,y)
plt.plot(x,model.predict(x))
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Linear Regression Marks")
plt.show()
