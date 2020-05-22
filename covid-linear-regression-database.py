import psycopg2
import pandas as pd
import numpy as np
from datetime import date, timedelta
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

def create_db(dbname, username, password, create):
    """ Creates/deletes the desired database
    
    Args:
        dbname (str): database name
        username (str): postgresql username with superuser permissions
        password (str): postgresql password for respective username
        create (bool): True for creation, False for deletion
    """

    # connect to superuser database
    conn = psycopg2.connect(database="postgres", user='bg', password='pass')
    conn.autocommit = True
    cur = conn.cursor()
    if (create):
        cur.execute('''CREATE DATABASE {}'''.format(dbname)) # create db
    else:
        cur.execute('''SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = 'mydb';''') # disconnect remaining user
        cur.execute('''DROP DATABASE {}'''.format(dbname)) # drop db
    
    # close database connection
    cur.close()
    conn.close()

def data_input(engine):
    """ Finds the desired data and inserts into database
    
    Args:
        engine (str): database engine
    """
    
    # setup database connection
    conn = engine.raw_connection()
    conn.autocommit = True
    cur = conn.cursor()
    
    v1_cols = ['Country/Region', 'Confirmed', 'Deaths', 'Recovered'] # format for up until 3/21 (inclusive)
    v2_cols = ['Country_Region', 'Confirmed', 'Deaths', 'Recovered'] # format for after 3/21

    delta = timedelta(days=1) # set to increment  1 day
    start = date(2020, 2, 1) # start date
    end = date.today() - delta # end date (non-inclusive), -1 day in case of checking before the daily update at 3:30-4:00 UTC
    
    # create empty table
    cur.execute('''CREATE TABLE daily_us(my_date DATE, active INT);''')

    while start < end:
        # gather data directly from github csv file
        url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/' + start.strftime('%m-%d-%Y') + '.csv'

        # read in data
        if start <= date(2020, 3, 21): # data format 1
            df = pd.read_csv(url, usecols=v1_cols)
        else: # data format 2
            df = pd.read_csv(url, usecols=v2_cols)    
        df['Active'] = df['Confirmed'] - df['Deaths'] - df['Recovered'] # calculate the active number of cases

        df = df.loc[df[df.columns[0]] == 'US'] # filter to only US data
        
        # add data to database table
        cur.execute('''INSERT INTO daily_us(my_date, active) VALUES (%(my_date)s,%(active)s);''', {'my_date':start, 'active':df['Active'].sum().item()})

        start += delta # increment date
    
    conn.commit()
    # close database connection
    cur.close()
    conn.close()

def data_processing_output(engine, train_test_split):
    """ Collects the data from the database and does analysis using linear regression
    
    Args:
        engine (str): database engine
    """
    
    # setup database connection
    conn = engine.raw_connection()
    cur = conn.cursor()

    delta = timedelta(days=1) # set to increment  1 day

    # Train/Test Split at the start of train_test_split (May = 5)
    # get training data from database
    cur.execute('''SELECT my_date FROM daily_us where EXTRACT(MONTH FROM my_date)<%(train_test_split)s;''',{'train_test_split':train_test_split})
    train_dates = cur.fetchall()
    train_dates = [tup[0] for tup in train_dates] # reformat data
    cur.execute('''SELECT active FROM daily_us where EXTRACT(MONTH FROM my_date)<%(train_test_split)s;''',{'train_test_split':train_test_split})
    y_train = cur.fetchall()
    y_train = [tup[0] for tup in y_train] # reformat data
    X_train = np.arange(len(y_train)).reshape(-1,1)

    # get test data from database
    cur.execute('''SELECT my_date FROM daily_us where EXTRACT(MONTH FROM my_date)>=%(train_test_split)s;''',{'train_test_split':train_test_split})
    test_dates = cur.fetchall()
    test_dates = [tup[0] for tup in test_dates] # reformat data
    cur.execute('''SELECT active FROM daily_us where EXTRACT(MONTH FROM my_date)>=%(train_test_split)s;''',{'train_test_split':train_test_split})
    y_test = cur.fetchall()
    y_test = [tup[0] for tup in y_test] # reformat data
    X_test = np.arange(len(y_train), len(y_test)+len(y_train)).reshape(-1,1)

    # close database connection
    cur.close()
    conn.close()

    # create and fit the linear regression model
    reg = LinearRegression().fit(X_train.reshape(-1,1), y_train)
    y_pred = reg.predict(X_test) # predict values
    report = pd.DataFrame(data = {'Date':test_dates, 'Actual':y_test, 'Predicted':y_pred.astype(int)}) # makes a compact report

    # create the visual
    plt.scatter(train_dates, y_train, c='b')
    plt.scatter(test_dates, y_test, c='g')
    plt.plot(test_dates, y_pred, color='k')

    plt.xlim((train_dates[0]-delta, test_dates[-1]+delta))
    plt.xticks(np.arange(train_dates[0]-delta, test_dates[-1]+delta, 25*delta))

    plt.legend(['Predicted Data (Linear Regression)', 'Training Data (2/1-4/31)', 'Testing Data (5/1-Present)'])

    plt.xlabel('Date')
    plt.ylabel('Active Infections')
    plt.title('Predicting Active US Covid-19 Infections using Linear Regression')
    plt.show()

    print('R^2 Value:', reg.score(X_test, y_test)) # scoring metric
    print('\nReport:\n',report.to_string(index=False))
    
def main():
    username = 'bg' # database username
    passw = 'pass' # database password
    dbname = 'mydb' # database name
    train_test_split = 5 # month to create train/test split (numeric form)

    # create the database
    create_db(dbname, username, passw, True)
 
    # connect to database
    engine = create_engine('postgresql+psycopg2://'+username+':'+passw+'@localhost/'+dbname)
    # take in data
    data_input(engine)
    # analyze the data
    data_processing_output(engine, train_test_split)
    
    # delete the database
    create_db(dbname, username, passw, False)

if __name__ == "__main__":
    main()
