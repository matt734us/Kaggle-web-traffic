
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
import warnings
import scipy
from datetime import timedelta

# For marchine Learning Approach
from statsmodels.tsa.tsatools import lagmat
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('fivethirtyeight')

warnings.filterwarnings('ignore')

# Load the data
train = pd.read_csv("./Documents/workspace/webTrafFor/train_1.csv")


# In[15]:

print(train.shape)


# In[1]:

#Pivots data to covert to transactional
train_flattened = pd.melt(train[list(train.columns[-50:])+['Page']], id_vars='Page', var_name='date', value_name='Visits')
#Date to datetime dtype
train_flattened['date'] = train_flattened['date'].astype('datetime64[ns]')
#New field to indicate weekend as 1
train_flattened['weekend'] = ((train_flattened.date.dt.dayofweek) // 5 == 1).astype(float)

# Median by page
df_median = pd.DataFrame(train_flattened.groupby(['Page'])['Visits'].median())
df_median.columns = ['median']

# Average by page
df_mean = pd.DataFrame(train_flattened.groupby(['Page'])['Visits'].mean())
df_mean.columns = ['mean']

# Merging data
train_flattened = train_flattened.set_index('Page').join(df_mean).join(df_median)

#reset index
train_flattened.reset_index(drop=False,inplace=True)

#Add column for weekday, day, month and year
train_flattened['weekday'] = train_flattened['date'].dt.dayofweek
train_flattened['day'] = train_flattened['date'].dt.day
train_flattened['month'] = train_flattened['date'].dt.month
train_flattened['year'] = train_flattened['date'].dt.year


# In[2]:

plt.figure(figsize=(50, 8))
#plot of mean number of visits per page by date
mean_group = train_flattened[['Page','date','Visits']].groupby(['date'])['Visits'].mean()
plt.plot(mean_group)
plt.title('Time Series - Average')
plt.show()


# In[3]:

plt.figure(figsize=(50, 8))
#plot of median number of visits per page by date
median_group = train_flattened[['Page','date','Visits']].groupby(['date'])['Visits'].median()
plt.plot(median_group, color = 'r')
plt.title('Time Series - median')
plt.show()


# In[4]:

plt.figure(figsize=(50, 8))
#plot of standard deviation of visits per page by date
std_group = train_flattened[['Page','date','Visits']].groupby(['date'])['Visits'].std()
plt.plot(std_group, color = 'g')
plt.title('Time Series - std')
plt.show()


# In[5]:

#move weekday and month integer values to a new column and rename current columns with a better label for reading
train_flattened['month_num'] = train_flattened['month']
train_flattened['month'].replace('11','11 - November',inplace=True)
train_flattened['month'].replace('12','12 - December',inplace=True)

train_flattened['weekday_num'] = train_flattened['weekday']
train_flattened['weekday'].replace(0,'01 - Monday',inplace=True)
train_flattened['weekday'].replace(1,'02 - Tuesday',inplace=True)
train_flattened['weekday'].replace(2,'03 - Wednesday',inplace=True)
train_flattened['weekday'].replace(3,'04 - Thursday',inplace=True)
train_flattened['weekday'].replace(4,'05 - Friday',inplace=True)
train_flattened['weekday'].replace(5,'06 - Saturday',inplace=True)
train_flattened['weekday'].replace(6,'07 - Sunday',inplace=True)


# In[6]:

#Create dataframe for heatmap of day of the week by the months of November and December
train_group = train_flattened.groupby(["month", "weekday"])['Visits'].mean().reset_index()
train_group = train_group.pivot('weekday','month','Visits')
train_group.sort_index(inplace=True)

sns.set(font_scale=3.5) 

# Draw a heatmap with the numeric values in each cell
f, ax = plt.subplots(figsize=(50, 30))
sns.heatmap(train_group, annot=False, ax=ax, fmt="d", linewidths=2)
plt.title('Web Traffic Months cross Weekdays')
plt.show()


# In[7]:

#Create dataframe for heatmap of day by the months of November and December
train_day = train_flattened.groupby(["month", "day"])['Visits'].mean().reset_index()
train_day = train_day.pivot('day','month','Visits')
train_day.sort_index(inplace=True)
train_day.dropna(inplace=True)

# Draw a heatmap with the numeric values in each cell
f, ax = plt.subplots(figsize=(50, 30))
sns.heatmap(train_day, annot=False, ax=ax, fmt="d", linewidths=2)
plt.title('Web Traffic Months cross days')
plt.show()


# In[8]:

#Create dataframe from mean visits per day group created above
times_series_means =  pd.DataFrame(mean_group).reset_index(drop=False)

#Add column for weekday, day, month and year
times_series_means['weekday'] = times_series_means['date'].dt.dayofweek
times_series_means['day'] = times_series_means['date'].dt.day
times_series_means['month'] = times_series_means['date'].dt.month
times_series_means['year'] = times_series_means['date'].dt.year


# In[9]:

times_series_means.reset_index(drop=True, inplace=True)

#Create function to create a number of values of previous data
def lag_func(data,lag):
    lagged = data.copy()
    for c in range(1,lag+1):
        lagged["lag%d" % c] = lagged['diff'].shift(c)
    return lagged

#Create function to get the difference between current number of visits and previous ones
def diff_creation(data):
    data["diff"] = np.nan
    data.ix[1:, "diff"] = (data.iloc[1:, 1].as_matrix() - data.iloc[:len(data)-1, 1].as_matrix())
    return data

#Call diff_creation function
df_count = diff_creation(times_series_means)

# Creation of 7 features with function lag_func
lag = 7
lagged = lag_func(df_count,lag)
last_date = lagged['date'].max()
lagged = lagged.fillna(0.0)


# In[20]:

print(lagged)


# In[13]:

# Train Test split
def train_test(data_lag):
    #Create columns list for x
    xc = ["lag%d" % i for i in range(1,lag+1)] + ['weekday'] + ['day']
    split = 0.70
    #remove first set of rows where lag data is not complete for xt ad yt
    xt = data_lag[(lag+3):][xc]
    yt = data_lag[(lag+3):]["diff"]
    #Create index for split
    isplit = int(len(xt) * split)
    #Split and return data
    x_train, y_train, x_test, y_test = xt[:isplit], yt[:isplit], xt[isplit:], yt[isplit:]
    return x_train, y_train, x_test, y_test, xt, yt

#Apply train_test function
x_train, y_train, x_test, y_test, xt, yt = train_test(lagged)


# In[21]:

print(y_train)


# In[22]:

def modelisation(x_tr, y_tr, x_ts, y_ts, xt, yt, model):
    # Modelisation with all product
    #train model
    model.fit(x_tr, y_tr)
    #predict with model
    prediction = model.predict(x_ts)
    #calculate mean absolute error and r2 of model
    r2 = r2_score(y_ts.as_matrix(), model.predict(x_ts))
    mae = mean_absolute_error(y_ts.as_matrix(), model.predict(x_ts))
    print ("-----------------------------------------------")
    print ("r2 with 80% of the data to train:", r2)
    print ("mae with 80% of the data to train:", mae)
    print ("-----------------------------------------------")

    # Model with all data
    model.fit(xt, yt) 
    
    return model, prediction

#Define model
model =  AdaBoostRegressor(n_estimators = 5000, random_state = 42, learning_rate=0.01)

#Call modelisation function
clr, prediction  = modelisation(x_train, y_train, x_test, y_test, xt, yt, model)


# In[23]:

# Line plot of predicted and observed values
plt.style.use('ggplot')
plt.figure(figsize=(50, 5))
line_up, = plt.plot(prediction,label='Prediction')
line_down, = plt.plot(np.array(y_test),label='Reality')
plt.ylabel('Series')
plt.legend(handles=[line_up, line_down])
plt.title('Performance of predictions - Benchmark Predictions vs Reality')
plt.show()


# In[24]:

# Plot scatter plot of predicted values against observed
fig, ax = plt.subplots(figsize=(50, 8))
plt.scatter(y_test.as_matrix(), prediction, color='b')
plt.title('Scatterplot Prediction vs Reality')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.show()


# In[25]:

# create function to create dataframe for dates that will be predicted and there corresponding day of the week
def pred_df(data,number_of_days):
    data_pred = pd.DataFrame(pd.Series(data["date"][data.shape[0]-1] + timedelta(days=1)),columns = ["date"])
    for i in range(number_of_days):
        inter = pd.DataFrame(pd.Series(data["date"][data.shape[0]-1] + timedelta(days=i+2)),columns = ["date"])
        data_pred = pd.concat([data_pred,inter]).reset_index(drop=True)
    return data_pred

data_to_pred = pred_df(df_count,30)


# In[28]:

def initialisation(data_lag, data_pred, model, xtrain, ytrain, number_of_days):
    #print(data_lag)
    # Initialisation
    model.fit(xtrain, ytrain)
    
    for i in range(number_of_days-1):
        #produce lag data for day predicting
        lag1 = data_lag.tail(1)["diff"].values[0]
        lag2 = data_lag.tail(1)["lag1"].values[0]
        lag3 = data_lag.tail(1)["lag2"].values[0]
        lag4 = data_lag.tail(1)["lag3"].values[0]
        lag5 = data_lag.tail(1)["lag4"].values[0]
        lag6 = data_lag.tail(1)["lag5"].values[0]
        lag7 = data_lag.tail(1)["lag6"].values[0]
#        lag8 = data_lag.tail(1)["lag7"].values[0]
        
        #Get day of week from date of data to predict
        data_pred['weekday'] = data_pred['date'].dt.dayofweek
        data_pred['day'] = data_pred['date'].dt.day
        #Get weekday as variable
        weekday = data_pred['weekday'][0]
        day = data_pred['day'][0]
        
        data = {'lag1': lag1,
                'lag2': lag2,
                'lag3': lag3,
                'lag4': lag4,
                'lag5': lag5,
                'lag6': lag6,
                'lag7': lag7,
#                'lag8': lag8,
               'weekday': weekday,
               'day': day}
        
        #Create array for lag data
        #row = pd.Series([lag1, lag2, lag3, lag4, lag5, lag6, lag7, lag8, weekday, day],
        #                ['lag1', 'lag2', 'lag3','lag4','lag5','lag6','lag7','lag8','weekday', 'day'])
        #Create empty dataframe
        to_predict = pd.DataFrame(data=data, index=[0])
#        print(to_predict)
        #Create emoty dataframe with diff column
        #prediction = pd.DataFrame(columns = ['diff'])
        #Append row to to_predict dataframe
#        to_predict = to_predict.append([row])

        #Use model to produce a prediction for current line
        prediction = pd.DataFrame(model.predict(to_predict),columns = ['diff'])
#        print(prediction)
        # Use calcualted diff value to calculate number of visits
        #if i == 0:
        last_predict = data_lag["Visits"][data_lag.shape[0]-1] + prediction.values[0][0]

        #if i > 0 :
#        last_predict = data_lag["Visits"][data_lag.shape[0]-1] + prediction.values[0][0]
        #print(last_predict)
        
        data_lag = pd.concat([data_lag, prediction.join(data_pred["date"]).join(to_predict)]).reset_index(drop=True)
#        print(data_lag)
        data_lag["Visits"][data_lag.shape[0]-1] = last_predict
        print(data_lag.shape)
        # test
        data_pred = data_pred[data_pred["date"]>data_pred["date"][0]].reset_index(drop=True)
#        print(data_pred)
    return data_lag

model_fin = AdaBoostRegressor(n_estimators = 5000, random_state = 42, learning_rate=0.01)

lagged = initialisation(lagged, data_to_pred, model_fin, xt, yt, 30)

lagged[lagged['diff']<0]
lagged.ix[(lagged.Visits < 0), 'Visits'] = 0


# In[29]:

#Plot of observed data and predicted data points
df_lagged = lagged[['Visits','date']]
df_train = df_lagged[df_lagged['date'] <= last_date]
df_pred = df_lagged[df_lagged['date'] >= last_date]
plt.style.use('ggplot')
plt.figure(figsize=(30, 5))
plt.plot(df_train.date,df_train.Visits)
plt.plot(df_pred.date,df_pred.Visits,color='b')
plt.title('Training time series in red, Prediction on 30 days in blue -- ML Approach')
plt.show()


# In[ ]:

print(lagged)


# In[ ]:



