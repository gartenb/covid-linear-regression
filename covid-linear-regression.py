import pandas as pd
import numpy as np
from datetime import date, timedelta
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

v1_cols = ['Country/Region', 'Confirmed', 'Deaths', 'Recovered'] # format for up until 3/21 (inclusive)
v2_cols = ['Country_Region', 'Confirmed', 'Deaths', 'Recovered'] # format for after 3/21

train_test_split = 5 # month to create train/test split (numeric form)

delta = timedelta(days=1) # set the increment to 1 day
start = date(2020, 2, 1) # start date
end = date.today() - delta # end date (non-inclusive), -1 day in case of checking before the daily update at 3:30-4:00 UTC


daily_us = pd.DataFrame(columns = ['Date', 'Active']) # empty dataframe


while start < end:
    # gather data directly from github csv file
    url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/' + start.strftime('%m-%d-%Y') + '.csv'
    
    # read in data
    if start <= date(2020, 3, 21): # data format 1
        df = pd.read_csv(url, usecols=v1_cols)
    else: # data format 2
        df = pd.read_csv(url, usecols=v2_cols)    
    df['Active'] = df['Confirmed'] - df['Deaths'] - df['Recovered']

    df = df.loc[df[df.columns[0]] == 'US'] # filter to only US data
    
    daily_us.loc[len(daily_us)] = pd.Series({'Date':pd.to_datetime(start), 'Active':df['Active'].sum()}) # add date + confirmed US cases to dataframe
    
    start += delta # increment date


# Train/Test Split at the start of train_test_split (May = 5)
train_dates = daily_us[daily_us['Date'].dt.month < train_test_split]['Date']
y_train = daily_us[daily_us['Date'].dt.month < train_test_split]['Active']
X_train = np.arange(len(y_train)).reshape(-1,1)

test_dates = daily_us[daily_us['Date'].dt.month >= train_test_split]['Date']
y_test = daily_us[daily_us['Date'].dt.month >= train_test_split]['Active']
X_test = np.arange(len(y_train), len(y_test)+len(y_train)).reshape(-1,1)

# create and fit the linear regression model
reg = LinearRegression().fit(X_train.reshape(-1,1), y_train)
y_pred = reg.predict(X_test) # predict values
report = pd.DataFrame(data = {'Date':test_dates, 'Actual':y_test, 'Predicted':y_pred.astype(int)}) # makes a compact report

# create the visual
plt.scatter(train_dates, y_train, c='b')
plt.scatter(test_dates, y_test, c='g')
plt.plot(test_dates, y_pred, color='k')

plt.xlim((daily_us.iloc[0,0]-delta, daily_us.iloc[-1,0]+delta))
plt.xticks(np.arange(daily_us.iloc[0,0]-delta, daily_us.iloc[-1,0]+delta, 25*delta))

plt.legend(['Predicted Data (Linear Regression)', 'Training Data (2/1-4/31)', 'Testing Data (5/1-Present)'])

plt.xlabel('Date')
plt.ylabel('Active Infections')
plt.title('Predicting Active US Covid-19 Infections using Linear Regression')
plt.show()

print('R^2 Value:', reg.score(X_test, y_test)) # scoring metric
print('\nReport:\n',report)
