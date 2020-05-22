## __Using Linear Regression to Predict Covid-19 Active Infections__

__Files: Description + How To Run__<br>
1. [covid-linear-regression-database.py](https://github.com/gartenb/covid-linear-regression/blob/master/covid-linear-regression-database.py)
	* Useses database functionality with Scikit Learn to create a prediction for Covid-19 with Linear Regression
	* Must have PostgreSQL installed. Make sure there is a User that can create databases [(documentation)](https://www.postgresql.org/docs/current/app-createuser.html). In the Main, change 'user', 'passw' and 'dbname' to your chosen PostgreSQL username, password and database name respectively (database name should not be associated with an existing database). In terminal, 'cd' into the correct forlder, then use execute the command:<br>
		$ python covid-linear-regression-database.py<br>
2. [covid-linear-regression.py](https://github.com/gartenb/covid-linear-regression/blob/master/covid-linear-regression.py)
	* This is a version of the previous program, without database functionality.
	* To run the program, in terminal 'cd' into the correct forlder, then use execute the command:<br>
		$ python covid-linear-regression.py<br>

Note: All Data Courtesy of [COVID-19 Data Repository by the Center for Systems Science and Engineering (CSSE) at Johns Hopkins University](https://github.com/CSSEGISandData/COVID-19)
