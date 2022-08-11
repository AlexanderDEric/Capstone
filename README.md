# Capstone

The goal of this Data Product is to Forecast a Turnover Rate of a organization or company

## Description

Data for this example has been provided by a Fortune 500 company after being aggregated to a Turnover Rate. Data is collected by manual entry of information into a Human Resources Information System (HRIS) by employees or members of the organization. This includes Hire or Termination Dates, Job Profile, Compensation, Location, Department, and many more attributes. From the HRIS, reports are built to expose different data regarding the above attributes or other Business Processes including Promotions, Transfers, Pay Raises, and other relevant information pertaining when attributes were modified or changed. 

These reports then are fed into a Data Warehouse using Python with little transformations applied. Data from HRIS reports are structured to be ready for analysis by the Business Process, Attribute, or Object. Each report lives in its own unique table, for example the Business Process table shows when Promotions, Transfers, or Pay Raises took place along with who submitted the request, when it was approved, when it went into effect, and more. This Business Process table can then be joined to the Employee table (Which is fed from a different HRIS report) that contains all most up to date information for each Employee.

<p align="center">
  <img src="https://user-images.githubusercontent.com/93106833/182761946-358a080a-3974-433c-90ae-77fb0def02d5.png" width="1000">
</p>

Because data is pulled from the HRIS in a formatted matter, little data manipulation is needed before feeding into a Statistical Model. To easily sum up the number of active employees within a time period, another table is built to show a period of dates (I.e., Jan 1st, Feb 1st, Mar 1st or Jan 1st, Jan 8th, Jan 15th, Jan 21st) along with the Employee ID and another two fields titled ‘IsActiveHC’ and ‘IsTerminatedHC’ to indicate whether an individual was actively working within the observed period date. Turnover Rate is then Calculated as the Number of Terminations / Number of Active Employees.

Before feeding data into the ARIMA model that is used for Predicting Turnover, exponential smoothing will be applied to create a stationary dataset that is easier to forecast with. However, given that exponential smoothing is a technique specific to ARIMA models, a detailed process will be covered in the Exploratory Analysis step.

<p align="center">
  <img src="https://user-images.githubusercontent.com/93106833/182762050-0abb2241-5b6a-4a12-bcb5-0ad1d6eb201d.png" width="500">
  <img src="https://user-images.githubusercontent.com/93106833/182762060-3a2403ca-8c78-427d-aacb-de0c1701a229.png" width="500">
</p>

After the stationary dataset is run through an ARIMA model, if exponential smoothing has been applied prior to running the model, the inverse function will need to be applied to convert the output values into actual Turnover values. Once actual Turnover values have been obtained, visualize of the final dataset results can be plotted to show the visual pattern of previous patterns and trend to the Forecasted Turnover values.

Finalized results should leave little room for interpretation. A confidence interval can be included in the finalized visual to indicate whether the prediction for a specific date is of high or low confidence. In the example above, a confidence interval might be wider during the tenth through twentieth week, while a tighter interval will occur during the latter part of the year.

<p align="center">
  <img src="https://user-images.githubusercontent.com/93106833/182762148-ffa5c619-3540-46bd-ac9e-3535cd48921c.png" width="1000">
</p>

Room for interpretation is left within the action that proceeds delivery of the information. If turnover is being predicted during a period of growth in the company, then results may be actioned on more directly to stay ahead of the potential number of terminations. If risk of the organization being overstaffed is present and no growth is occurring, then leadership may ignore the number of terminations being forecasted as the schedule to backfill the positions may not be as prioritized. 

## Getting Started

To get started, download the 'DSC-580 Week 6.py' file and upload your own data. Source code has been written so that you can upload an excel worksheet using Pandas, or read into a SQL Database using PYODBC. Data must be by the week, with Monday being the start of a numbered Date in your dataset. I.e. my first Dated record in the example dataset is 2019-01-14, as Jan 14th in 2019 was a Monday.

### Dependencies

Assuming a dataset has been cleaned and ready to be fed into the Python script, the following software and libraries are required to deploy the entire software from Running the model to Visualization:

- Python 3.9+
  - Source code was written in an Anaconda environment
- Python Libraries 
  - Urllib
  - Pyodbc
  - Sqlalchemy
  - Numpy
  - Pandas
  - Matplotlib.pyplot
  - Itertools
  - Sklearn.metrics
  - Statsmodels.tsa.arima_model
- Data Storage Software (Any of the following or others will work):
  - A URL with a CSV file
  - SQL Database
  - Excel
  - HRIS System
- Tableau (Or other Data Visualization tool)


### Installing

Download the 'DSC-580 Week 6.py'

### Executing program

Funtional Requirements


```
Forecast Turnover Rate   -     CreateFinalTable()
Plot Results             -     PredictTurnover()
Generate Report          -     WriteToExcel()
```

Functions in the source code has been nested, so only one function needs to be ran to get your desired output.

```
ReadSQL
- Will read your data in from a SQL Database
ReadExcel
- Will read your data in from an Excel file
ExpSmooth
- Begins the process of smoothing your data set
OptimizeArima
- Select the best Model Paramteres to use
RunArima
- Runs the ARIMA model based on model paramteres selected above
Predictions
- Obtain predictions over past values and explain variance
CreateForecast
- Obtain forecasted values from ARIMA model
InvExp
- If ExpSmooth was applied, inverse the forecasted results back into actual
r2score
- Create r2Score
CreateFinalTable
- Creates final report to be exported
PredictTurnover
- Plots finalizaed graph out outputted values
WriteToSql
- Will save data to a SQL table
WriteToExcel
- Will save data to an Excel file
```

## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
```

## Authors

Contributors names and contact info

ex. Eric Alexander
ex. https://github.com/AlexanderDEric

## Version History

* 0.1
    * Initial Release

## Acknowledgments

- Grand Canyon University
- DSC-580: Designing and Creating Data Products
- Dr. Michelle Bennett
