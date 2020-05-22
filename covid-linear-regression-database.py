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

def data_input(conn, cur):
    """ Finds the desired data and inserts into database
    
    Args:
        conn (Connection): connection to the database
        cur (Cursor): cursor to the database
    """

    conn.autocommit = True
    
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

def data_retrieval(cur, train, train_test_split, start_i = 0):
    """ Collects and reformats the data from the database
    
    Args:
        cur (Cursor): cursor to the database
        train (bool): True for Training, False for Testing
        train_test_split (str): the train test split (May = 5)
        start_i (int): defaults to 0, used when looking at test data to pre-increment its indeces
        
    Return:
        dates ([datetime.date]): the sequential dates
        X ([int]): an index corresponding to each date
        y ([int]): the active number of infections at corresponding given date
    """

    delta = timedelta(days=1) # set to increment  1 day
    if (train):
        inequality = '<'
    else:
        inequality = '>='
    
    # get data from database
    cur.execute('''SELECT my_date FROM daily_us where EXTRACT(MONTH FROM my_date){}{};'''.format(inequality, train_test_split))
    dates = cur.fetchall()
    dates = [tup[0] for tup in dates] # reformat data
    cur.execute('''SELECT active FROM daily_us where EXTRACT(MONTH FROM my_date){}{};'''.format(inequality, train_test_split))
    y = cur.fetchall()
    y = [tup[0] for tup in y] # reformat data
    X = np.arange(start_i,start_i+len(y)).reshape(-1,1) # create sequential indeces

    return dates, X, y

def lin_reg(X_train, X_test, y_train, y_test):
    """ Creates, fits and scores the linear regression model
    
    Args:
        test_dates ([datetime.date]): list of dates for test data
        X_train ([int]): training data indeces
        X_test ([int]): test data indeces
        y_train ([int]): training data of active infections
        y_test ([int]): test data of active infections
        
    Return: 
        y_pred ([int]): predicted data of active infections for test_dates
    """

    reg = LinearRegression().fit(X_train, y_train)
    y_pred = reg.predict(X_test) # predict values
    print('R^2 Value:', reg.score(X_test, y_test)) # scoring metric
    
    return y_pred

def report(test_dates, y_test, y_pred):
    """ Creates a compact report of data and regression predictions
    
    Args:
        test_dates ([datetime.date]): list of dates for test data
        y_test ([int]): test data of active infections
        y_pred ([int]): predicted data of active infections for test_dates
    """

    report = pd.DataFrame(data = {'Date':test_dates, 'Actual':y_test, 'Predicted':y_pred.astype(int)})
    print('\nReport:\n',report.to_string(index=False))

def visual(train_dates, test_dates, y_train, y_test, y_pred):
    """ Visualizes data
    
    Args:
        train_dates ([datetime.date]): list of dates for training data
        test_dates ([datetime.date]): list of dates for test data
        y_train ([int]): training data of active infections
        y_test ([int]): test data of active infections
        y_pred ([int]): predicted data of active infections for test_dates
    """
    
    delta = timedelta(days=1) # set to increment  1 day
    
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


def main():
    username = 'bg' # database username
    passw = 'pass' # database password
    dbname = 'mydb' # database name
    train_test_split = 5 # month to create train/test split (numeric form)

    # create the database
    create_db(dbname, username, passw, True)
 
    # connect to database
    engine = create_engine('postgresql+psycopg2://'+username+':'+passw+'@localhost/'+dbname)
    conn = engine.raw_connection()
    cur = conn.cursor()
    
    # take in data
    data_input(conn, cur)
    
    # retrieve the data
    train_dates, X_train, y_train = data_retrieval(cur, True, train_test_split)
    test_dates, X_test, y_test = data_retrieval(cur, False, train_test_split, len(X_train))
    
    # model the data
    y_pred = lin_reg(X_train, X_test, y_train, y_test)
    
    # create a results report
    report(test_dates, y_test, y_pred)
    
    # create visualizations
    visual(train_dates, test_dates, y_train, y_test, y_pred)
    
    # close database connection
    cur.close()
    conn.close()
    
    # delete the database
    create_db(dbname, username, passw, False)

if __name__ == "__main__":
    main()
