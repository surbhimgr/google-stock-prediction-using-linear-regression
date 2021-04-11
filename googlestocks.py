# importing different packages
import time
import math
import quandl
import pickle #converts python object into stream of bytes so that it can be stored into memory
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, cross_validation, svm

# style of plot
style.use('ggplot')

# getting data
df = quandl.get("SSE/GGQ1", authtoken="pJyV7Md_sQnXWV2dyhP7")

# adding columns in df
#deviation in stock price
df['HL_PCT'] = ((df['High'] - df['Last'])/df['Last']) * 100.0
df['PCT_CHANGE'] = ((df['Last'] - df['Previous Day Price'])/df['Previous Day Price']) * 100.0

df = df[['Last', 'HL_PCT', 'PCT_CHANGE', 'Volume']]


forecast_col = 'Last'
df.fillna(-99999, inplace=True) #fill empty places

forecast_out = int(15)
print (forecast_out)

#shifting indexes
df['label'] = df[forecast_col].shift(-forecast_out)

#first 5 rows of df
print (df.head())

#defining x-axis
X = np.array(df.drop(['label', 'Last'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

#remove missing values
df.dropna(inplace=True)
y = np.array(df['label']) #y-axis

print (len(X), len(y))

#splitting into train/test models
#test_size=0.2 refers to 20% of the data is used for testing
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = LinearRegression(n_jobs=-1) #using all processors
clf.fit(X_train, y_train) #trainig x and y sets

#opening a file
with open('linearregression.pickle', 'wb') as file:
	pickle.dump(clf, file)

pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in) #to unserialise a data stream

accuracy = clf.score(X_test, y_test)
forecast_set = clf.predict(X_lately)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = time.mktime(datetime.datetime.strptime(str(last_date), "%Y-%m-%d %H:%M:%S").timetuple())

one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
	next_date = datetime.datetime.fromtimestamp(next_unix)
	next_unix += one_day

	df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

print (df.head())
print ("")
print (df.tail())
print ("")

df['Last'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')

print ("Forecast: {} \n Accuracy: {} Days: {}".format(forecast_set, accuracy*100, forecast_out))

plt.show()
