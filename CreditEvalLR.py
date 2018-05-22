
#This is the program to find the regression coefficients of the given data and also to predict the outcomes, in order to get a regression equation 
#The dependent variables are numerically stored in the MarksX matrix, while the experimented(rather random outputs) stored in the CreditY, in order to implement a credit evaluation system of performance by students

from sklearn import linear_model

MarksX = [[24,29], [34,42], [21,19], [27,36], [38,21], [23,19], [12,16], [19,40], [16,20], [37,23], [24, 45], [20,18], [45, 30], [42, 37], [28, 24], [26,29], [25, 17], [39, 43], [41, 46], [48, 33]]

CreditY = [6,8,3,8,5,3,3,9,4,3,9,3,6,7,5,5,2,9,9,7]

regr = linear_model.LinearRegression()
regr.fit(MarksX, CreditY)

Y_predict = regr.predict(MarksX)

print(Y_predict)

print("Original ",regr.coef_)


print(regr.intercept_)
