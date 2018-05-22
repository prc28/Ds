#This is the program using the RidgeCV method
#In this there's an array of all possible ridges is used, and the cross-validation facility selects the most suitable ridge in order to reach a state of colinearity in the relation infographics.



from sklearn import linear_model

MarksX = [[24,29], [34,42], [21,19], [27,36], [38,21], [23,19], [12,16], [19,40], [16,20], [37,23], [24, 45], [20,18], [45, 30], [42, 37], [28, 24], [26,29], [25, 17], [39, 43], [41, 46], [48, 33]]

CreditY = [6,8,3,8,5,3,3,9,4,3,9,3,6,7,5,5,2,9,9,7]

regr = linear_model.RidgeCV(alphas=[0.1, 0.2, 0.5, 0.9, 1.0, 10, 20, 21, 25, 30, 31, 32, 33, 34, 34.1, 34.2, 34.3, 34.4, 34.5, 35, 38, 40, 41, 45])
regr.fit(MarksX, CreditY)

Y_predict = regr.predict(MarksX)

#print(Y_predict)
print(regr.coef_)

#regr.fit(MarksX, Y_predict)

#print("Revised ", regr.coef_)
print(regr.intercept_)

print(regr.alpha_)
