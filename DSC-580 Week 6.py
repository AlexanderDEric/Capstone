#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Depending on how you are feeding data into the source code below, please refer to the Pandas or PYODBC libraries
#for information on which file type or SQL Database arguments to include.


# In[1]:


import warnings
warnings.filterwarnings('ignore')

import pyodbc
import pandas as pd

sql_conn = pyodbc.connect('DRIVER={***};SERVER=***;Trusted_Connection=***') 
query = '''

SELECT ID, Date, EmployeeId, JobProfile, JobFamily, CostCenter, WorkerType, ManagementLevel, TimeType, 
Location, LocationState, HireDate, TerminationDate, LengthOfServiceInMonthsFromHireDate AS Tenure,
IsActiveHC, IsTerminatedHC

FROM PeopleOps.workday.FactWeeklyHC

'''

df = pd.read_sql(query, sql_conn)


# In[1]:


get_ipython().run_line_magic('reset_selective', '-f ^(?!(df)$).*$')

import warnings
warnings.filterwarnings('ignore')

import os
import time, datetime
import urllib
import pyodbc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from itertools import product
from sklearn import metrics
from statsmodels.tsa.arima_model import ARIMA

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = [12, 6]

df0 = pd.DataFrame()


# In[19]:


#class ForecastTurnoverWeekly:    
    
#    def __init__(self, dataframe, variable, feature):
#        self.dataframe = dataframe
#        self.variable = variable
#        self.feature = feature


    
def GetTurnoverRate(dataframe, variable):

    dataframe['Date'] = pd.to_datetime(dataframe['Date'])

    df_sum = dataframe.groupby(['Date',variable])['IsActiveHC'].sum().reset_index()
    df_sum2 = dataframe.groupby(['Date',variable])['IsTerminatedHC'].sum().reset_index()

    df_sum = df_sum.merge(df_sum2)    
    df_sum['TurnoverRate'] = round((df_sum['IsTerminatedHC'] / df_sum['IsActiveHC'])*100, 3)

    return df_sum

def GetFeatureFrame(dataframe, variable, feature):

    df_sum = GetTurnoverRate(dataframe, variable)

    VarNames = []
    VarFrames = {}

    for v in df_sum[variable].unique():
        VarNames.append(v)

    for VName in VarNames:
        cond = VName == df_sum[variable]
        rows = df_sum.loc[cond, :]
        VarFrames[VName] = pd.DataFrame(rows)

    VarFrames[feature] = VarFrames[feature].drop(variable, axis=1)
    VarFrames[feature] = VarFrames[feature].set_index('Date')

    return VarFrames[feature]

def ExpSmooth(dataframe, metric, variable, feature):

    global df0

    df_feature = GetFeatureFrame(dataframe, variable, feature)

    df_feature[f'{metric}_log'] = np.log(df_feature[metric])    
    df_feature[f'{metric}_shift'] = df_feature[f'{metric}_log'].shift()
    df_feature[f'{metric}_shift_diff'] = df_feature[f'{metric}_log'] - df_feature[f'{metric}_shift']
    df_feature = pd.DataFrame(df_feature.iloc[1:-1])

    df_feature = df_feature.reset_index()
    df_feature['Week Num'] = pd.to_datetime(df_feature['Date']).dt.week
    weekly_avg = df_feature.groupby(df_feature['Week Num'])[f'{metric}_log'].mean()
    df_feature['Month Num'] = pd.to_datetime(df_feature['Date']).dt.month
    month_avg = df_feature.groupby(df_feature['Month Num'])[f'{metric}_log'].mean()
    df_feature['Quarter Num'] = pd.to_datetime(df_feature['Date']).dt.quarter
    quarter_avg = df_feature.groupby(df_feature['Quarter Num'])[f'{metric}_log'].mean()
    df_feature['Year'] = pd.to_datetime(df_feature['Date']).dt.year

    df_feature = df_feature.join(quarter_avg, on='Quarter Num', how='left', lsuffix='_left', rsuffix='_right')
    df_feature = df_feature.drop([f'{metric}_log_left'], axis=1)
    df_feature = df_feature.rename(columns={f'{metric}_log_right': 'Quarter Avg'})
    df_feature = df_feature.join(month_avg, on='Month Num', how='left', lsuffix='_left', rsuffix='_right')
    df_feature = df_feature.join(month_avg, on='Month Num', how='left', lsuffix='_left', rsuffix='_right')
    df_feature = df_feature.drop([f'{metric}_log_left'], axis=1)
    df_feature = df_feature.rename(columns={f'{metric}_log_right': 'Month Avg'})
    df_feature = df_feature.join(weekly_avg, on='Week Num', how='left', lsuffix='_left', rsuffix='_right')
    df_feature = df_feature.join(weekly_avg, on='Week Num', how='left', lsuffix='_left', rsuffix='_right')
    df_feature = df_feature.drop(f'{metric}_log_left', axis=1)
    df_feature = df_feature.rename(columns={f'{metric}_log_right': 'Week Avg'})

    df_feature['Weight 1'] = (df_feature['Quarter Avg'] + df_feature['Month Avg'] + df_feature['Week Avg'])/3

    if df0.empty:
        df0 = df0.append(df_feature)

    return df_feature

def OptimizeArima(dataframe, metric, variable, feature):

    ExpSmooth(dataframe, metric, variable, feature)

    ps = range(0, 8, 1)
    d = 1
    qs = range(0, 8, 1)

    order_list = [(0,0,0)]
    optimize_results = []

    parameters = product(ps, qs)
    parameters_list = list(parameters)

    for each in parameters_list:
        each = list(each)
        each.insert(1, 1)
        each = tuple(each)
        order_list.append(each)

    for order in order_list:
        try: 
            model = ARIMA(df0[f'{metric}_shift_diff'], order=order).fit(disp=-1)
        except:
            continue

        aic = model.aic
        optimize_results.append([order, model.aic])

    df_optimize = pd.DataFrame(optimize_results, columns=['(p, d, q)', 'AIC'])
    df_optimize = df_optimize.sort_values(by='AIC', ascending=True).reset_index(drop=True)

    return df_optimize

def RunArima(dataframe, metric, variable, feature):

    df_optimize = OptimizeArima(dataframe, metric, variable, feature)

    results = ARIMA(df0[f'{metric}_shift_diff'], order=df_optimize['(p, d, q)'][0]).fit(disp=-1)
    forecast = pd.DataFrame(results.forecast(52))
    forecast = forecast.T.rename(columns={0: 'Forecast', 1: 'StdErr', 2: 'Conf_Int'})

    return forecast

def Predictions(dataframe, metric, variable, feature):

    df_optimize = OptimizeArima(dataframe, metric, variable, feature)

    results = ARIMA(df0[f'{metric}_shift_diff'], order=df_optimize['(p, d, q)'][0]).fit(disp=-1)
    predict = pd.Series(results.predict(start=1, end=len(df0)))

    return predict

def CreateForecast(dataframe, metric, variable, feature):

    forecast = RunArima(dataframe, metric, variable, feature)

    ForecastDates = df0.iloc[-1]
    ForecastDates = pd.DataFrame(pd.date_range(start=ForecastDates['Date'], periods=52, freq='W-MON'))
    ForecastDates = ForecastDates.rename(columns={0: 'Date'})
    ForecastDates = ForecastDates.join(forecast)
    ForecastDates = ForecastDates.drop(['StdErr','Conf_Int'], axis=1)

    ForecastDates['Week Num'] = pd.to_datetime(ForecastDates['Date']).dt.week
    ForecastDates['Month Num'] = pd.to_datetime(ForecastDates['Date']).dt.month
    ForecastDates['Quarter Num'] = pd.to_datetime(ForecastDates['Date']).dt.quarter
    ForecastDates['Week Num'] = ForecastDates['Week Num'].astype('int64')
    ForecastDates['Month Num'] = ForecastDates['Month Num'].astype('int64')
    ForecastDates['Quarter Num'] = ForecastDates['Quarter Num'].astype('int64')

    weights = df0[['Week Num','Month Num','Quarter Num','Weight 1','Week Avg','Month Avg','Quarter Avg']]
    ForecastDates = pd.merge(ForecastDates, weights, how='inner', on=['Week Num','Month Num','Quarter Num'])
    ForecastDates = ForecastDates.drop_duplicates()
    ForecastDates = ForecastDates.set_index('Date')

    return ForecastDates

def InvExp(dataframe, metric, variable, feature):

    ForecastDates = CreateForecast(dataframe, metric, variable, feature)

    predictions_ARIMA = pd.Series(ForecastDates['Forecast'], copy=True)
    predictions_ARIMA_shift = predictions_ARIMA.shift(periods=-1)
    predictions_ARIMA_weight = pd.Series(ForecastDates['Weight 1'], index=ForecastDates.index)
    predictions_ARIMA_log = predictions_ARIMA_weight.add(predictions_ARIMA_shift, fill_value=0)
    predictions_ARIMA = pd.DataFrame(np.exp(predictions_ARIMA_log.astype('float64')), columns=['PredictedValue'])
    predictions_ARIMA = predictions_ARIMA.shift(periods=-1)
    ForecastTable = predictions_ARIMA.join(ForecastDates, how='left')
    ForecastTable = ForecastTable.merge(df0, how='left', on='Date')
    ForecastTable = ForecastTable[['Date','PredictedValue',f'{metric}']]

    return ForecastTable

def r2score(dataframe, metric, variable, feature):

    ARIMA_m = Predictions(dataframe, metric, variable, feature)

    ARIMA_shift = ARIMA_m.shift(periods=-1)
    ARIMA_log = pd.Series(np.log(df0[f'{metric}']), index=df0.index)
    ARIMA_log = ARIMA_log.add(ARIMA_shift, fill_value=0)
    ARIMA_m = pd.DataFrame(np.exp(ARIMA_log))
    ARIMA_m = ARIMA_m.shift(periods=-1)
    predtable = ARIMA_m.join(df0[f'{metric}'])
    predtable = predtable.rename(columns={0: 'PredictedValue'})
    predtable['PredictedValue'] = round(predtable['PredictedValue'], 3)
    predtable = predtable[:-1]
    predtable = predtable.dropna()

    Finalr2 = metrics.r2_score(predtable[f'{metric}'], predtable['PredictedValue'])

    return Finalr2

def CreateFinalTable(dataframe, metric, variable, feature):

    FinalTable = InvExp(dataframe, metric, variable, feature)
    Finalr2 = r2score(dataframe, metric, variable, feature)
    df_optimize = OptimizeArima(dataframe, metric, variable, feature)

    FinalTable['r2score'] = round(Finalr2, 3)
    FinalTable['YearNum'] = pd.DataFrame(pd.to_datetime(FinalTable['Date']).dt.year).astype('int64')
    FinalTable['WeekNum'] = pd.DataFrame(pd.to_datetime(FinalTable['Date']).dt.week).astype('int64')
    FinalTable['Feature'] = feature
    pdq = int(''.join(filter(str.isdigit, str(df_optimize['(p, d, q)'][0]))))
    FinalTable['Version'] = str(1) + str('.') + str(pdq)
    FinalTable['AIC'] = round(df_optimize['AIC'][0], 3)
    FinalTable['PredictedValue'] = round(FinalTable['PredictedValue'], 3)
    FinalTable['ScriptRunDate'] = datetime.datetime.today().strftime('%Y-%m-%d')
    FinalTable = FinalTable[['Date','YearNum','WeekNum','PredictedValue',f'{metric}','Feature','Version','AIC','r2score','ScriptRunDate']]
    FinalTable = FinalTable[0:-1]

    return FinalTable

def PredictTurnover(dataframe, metric, variable, feature):

    FinalTable = CreateFinalTable(dataframe, metric, variable, feature)

    groups = df0.groupby('Year')
    for name, group in groups:
        plt.title('Turnover Rate over time Grouped by Year')
        plt.plot(group['Week Num'], group[f'{metric}'], marker="o", linestyle="", label=name)

    groups2 = FinalTable.groupby('YearNum')
    for name2, group2 in groups2:
        plt.title('Turnover Rate over time Grouped by Year')
        plt.plot(group2['WeekNum'], group2['PredictedValue'], marker="o", linestyle="--", label=name2)

    plt.legend()
    plt.show();

def WriteToSql(dataframe, metric, variable, feature):

    FinalTable = CreateFinalTable(dataframe, metric, variable, feature)

    connStr = ('DRIVER={ODBC Driver 17 for SQL Server};SERVER=PEOPLEOPSTEST,1437;DATABASE=PeopleOps;Trusted_Connection=yes')
    quoted_conn_str = urllib.parse.quote_plus(connStr)
    engine = create_engine(f'mssql+pyodbc:///?odbc_connect={quoted_conn_str}', fast_executemany=True)

    FinalTable.to_sql('ForecastTurnoverWeekly', con = engine, schema = 'DataScience', if_exists='append', index=False)
    print('Done!')
    
def WriteToExcel(dataframe, metric, variable, feature):

    FinalTable = CreateFinalTable(dataframe, metric, variable, feature)

    df2.FinalTable(r'C:\Users\TurnoverRateResults.xlsx')
    print('Done!')


# In[21]:


PredictTurnover(df,'TurnoverRate','ManagementLevel','Hourly Employee')


# In[17]:


CreateFinalTable(df,'TurnoverRate','ManagementLevel','Hourly Employee')


# In[20]:


#WriteToSql(df,'TurnoverRate','ManagementLevel','Hourly Employee')


# In[ ]:




