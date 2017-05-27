import pandas as pd 
import numpy as np 
import quandl, math, datetime
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt 
from matplotlib import style 

style.use('ggplot')

api_key = 'yowkuu9SozBMsyY_-vZq'
df = quandl.get('WIKI/GOOGL', authtoken = api_key)
#print(df)

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

# Defining the high and low % change
df['HL_PCT'] = ((df['Adj. High']-df['Adj. Low'])/df['Adj. Low']) * 100

#Defining the %change of opening and closing price
df['PCT_change'] = ((df['Adj. Close']-df['Adj. Open'])/df['Adj. Open']) * 100
#print(df['PCT_change'])

## Dataframe containing our features to use in regression model
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

#print(df)
#forecast_out = 'Adj. Close'
forecast_col = 'Adj. Close'
df.fillna(-99999, inplace = True)
forecast_out = int(math.ceil(0.01*len(df)))
#print(forecast_out)

## Our prediction column
df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace = True)
y = np.array(df['label'])

## Defining our testing and training data using sklearn 
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.2)

##Defining the classifier, and the number of threads to be used to go through data
clf = LinearRegression(n_jobs = 1)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
#print("Model Accuracy: ", accuracy)

forcast_set = clf.predict(X_lately)

df['Forecast'] = np.nan

## Defining the time axis of prediction graph
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day


## Defining data set to predict only closing stock prices and defining everything else as NaN
for i in forcast_set:
	next_date = datetime.datetime.fromtimestamp(next_unix)
	next_unix += one_day
	df.loc[next_date] = [np.nan for w in range(len(df.columns)-1)] + [i] 

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc = 4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


