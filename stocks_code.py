#!/usr/bin/env python
# coding: utf-8

# In[150]:


import nsepy as nse
from datetime import date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from plotly.offline import iplot
import plotly as py 
import cufflinks as cff
from numpy import sqrt,log
from statsmodels.tsa.seasonal import seasonal_decompose 
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.holtwinters import SimpleExpSmoothing,Holt,ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima_model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import tensorflow
import warnings
warnings.filterwarnings("ignore")


# In[2]:


symbol1="reliance"
nse_rel1=nse.get_history(symbol1,start=date(2012,5,22),end=date(2022,5,22))


# In[3]:


nse_rel1.to_csv("relaince_data.csv")


# In[4]:


symbol2="tcs"
nse_rel2=nse.get_history(symbol2,start=date(2012,5,22),end=date(2022,5,22))


# In[5]:


nse_rel2.to_csv("tcs_data.csv")


# # Relaince Data

# In[2]:


df1=pd.read_csv("relaince_data.csv")
df1


# In[3]:


2427


# # TCS Data

# In[4]:


df2=pd.read_csv("tcs_data.csv")
df2


# In[4]:


df1.shape,df2.shape


# In[12]:


df1.info()


# In[13]:


df2.info()


# In[15]:


df1.describe()


# In[16]:


df2.describe()


# In[18]:


df1.isnull().sum()


# In[19]:


df2.isnull().sum()


# In[5]:


df1=df1.rename(columns={"Deliverable Volume":"Deliverablevolume","%Deliverble":"Deliverable"})
df2=df2.rename(columns={"Deliverable Volume":"Deliverablevolume","%Deliverble":"Deliverable"})


# In[220]:


df1.duplicated().sum(),df2.duplicated().sum().sum()


# In[23]:


print(plt.style.available)
plt.style.use('seaborn-deep')


# In[221]:


df_open=pd.DataFrame()
df_open["Relaince"]=pd.Series(df1["Open"])
df_open["TCS"]=pd.Series(df2["Open"])
df_open["Date"]=pd.Series(df1["Date"])


# In[25]:


plt.rcParams["figure.figsize"]=(15,10)
sns.pairplot(df1)


# In[222]:


# Relaince Variables
plt.figure(figsize=(15,6))
sns.heatmap(df1.corr(),annot=True)


# In[28]:


# TCS Variables
plt.figure(figsize=(15,6))
sns.heatmap(df2.corr(),annot=True)


# # Opening price of the stock when the market opens

# In[223]:


plt.style.use("dark_background")
plt.rcParams["figure.figsize"]=(22,12)
df_open.plot(x="Date")
plt.ylabel("Price",size=25)
plt.legend(loc="upper left")


# In[224]:


df_close=pd.DataFrame()
df_close["Relaince"]=pd.Series(df1["Close"])
df_close["TCS"]=pd.Series(df2["Close"])
df_close["Date"]=pd.Series(df1["Date"])


# In[225]:


plt.style.use("tableau-colorblind10")


# # Closing price of the stock when the market closed

# In[228]:


plt.rcParams["figure.figsize"]=(22,9)
df_close.plot(x="Date")
plt.ylabel("Price",size=25)
plt.legend(loc="upper left")


# In[229]:


df_volume=pd.DataFrame()
df_volume["Relaince"]=pd.Series(df1["Volume"])
df_volume["TCS"]=pd.Series(df2["Volume"])
df_volume["Date"]=pd.Series(df1["Date"])


# # Total amount of stock traded on that day

# In[230]:


# Volume 
plt.style.use("tableau-colorblind10")
df_volume.plot(x="Date")
plt.legend(loc="upper left")
plt.ylabel("Count",size=23)


# In[250]:


# To identify the Trend and Sesonality
df1_seasonal=seasonal_decompose(df1.Close,freq=50)

df1_seasonal.plot()
# ACF Plot - to identify the corrrlation 
plot_acf(df1["Close"],lags=30)

plt.show()
plt.rcParams["figure.figsize"]=(12,4)


# In[248]:


# To identify the Trend and Sesonality
df2_seasonal=seasonal_decompose(df2.Close,freq=50)
df2_seasonal.plot()
# ACF Plot - to identify the corrrlation 
plot_acf(df2["Close"],lags=30)

plt.show()
plt.rcParams["figure.figsize"]=(12,4)


# # Moving average

# In[37]:


# Moving Average - Relaince
plt.style.use("dark_background")
R_MA=pd.DataFrame()
R_MA["50"]=df1["Close"].rolling(50).mean()
R_MA["100"]=df1["Close"].rolling(100).mean()
R_MA["200"]=df1["Close"].rolling(200).mean()
R_MA["Date"]=df1["Date"]
R_MA["Close"]=df1["Close"]
R_MA=R_MA.set_index("Date")
label=["MA50","MA100","MA200","Relaince Close"]
plt.rcParams["figure.figsize"]=(15,7)
R_MA.plot()
plt.ylabel("Average")
plt.legend(label,loc="upper left")


# In[39]:


# Moving Average - TCS
T_MA=pd.DataFrame()
T_MA["50"]=df2["Close"].rolling(50).mean()
T_MA["100"]=df2["Close"].rolling(100).mean()
T_MA["Date"]=df2["Date"]
T_MA["Close"]=df2["Close"]
T_MA=T_MA.set_index("Date")
label=["MA50","MA100","TCS Close"]
T_MA.plot()
plt.ylabel("Average")
plt.legend(label,loc="upper left")


# In[5]:


py.offline.init_notebook_mode(connected=True)
cff.go_offline()


# In[6]:


# Reliance Year wise Average price
df1_Quarter=pd.read_csv("relaince_data.csv",parse_dates=True,index_col="Date")
df1_Quarter=df1_Quarter.Close.resample('Y').mean()

plt.rcParams["figure.figsize"]=(20,6)
df1_Quarter.iplot(kind="line",color="black")


# In[6]:


# TCS Year wise Average price
df2_Quarter=pd.read_csv("tcs_data.csv",parse_dates=["Date"],index_col="Date")
df2_Quarter=df2_Quarter.Close.resample('Y').mean()
df2_Quarter.iplot(kind="line",color="black")


# In[7]:


df_high=pd.DataFrame()
df_high["TCS"]=df2.High
# TCS high prices
# Reliance high prices
df_high["Reliance"]=df1.High
# Making Date as a Index 
df_high["Date"]=df1.Date
df_high=df_high.set_index("Date")
df_high.iplot(kind="bar")


# In[8]:


df_low=pd.DataFrame()
df_low["TCS"]=df2.Low
# TCS low prices
# Reliance low prices
df_low["Reliance"]=df1.Low
# Making Date as a Index 
df_low["Date"]=df1.Date
df_low=df_low.set_index("Date")
df_low.iplot(kind="bar")


# In[47]:


plt.style.use("dark_background")


# In[262]:


df_volatility=pd.DataFrame()
df_volatility["Reliance"]=(df1["Close"]/df1["Close"].shift(1))-1
df_volatility["TCS"]=(df2["Close"]/df2["Close"].shift(1))-1
df_volatility["Reliance"].hist(bins=200,alpha=1.0,label="Reliance")
df_volatility["TCS"].hist(bins=200,alpha=0.7,label="TCS")
plt.legend()


# In[49]:


df1["High"].max(),df1["Low"].max()


# In[50]:


df2["High"].max(),df2["Low"].max()


# In[25]:


# 2012-05-22 - 2022-05-20
plt.style.use("dark_background")
plt.rcParams["figure.figsize"]=(18,5)
H_max_x1 = [2856.15,2786.10]
L_max_y1 = ["Highest Price","Lowest Price"]

plt.barh(L_max_y1,H_max_x1)
plt.title("Reliance Stock - Maximum Highest and Lowest Price",size=25)
 
for index, value in enumerate(H_max_x1):
    plt.text(value,index,
             str(value))


# In[54]:


plt.style.use("classic")
plt.rcParams["figure.figsize"]=(18,4)
H_max_x2 = [4043.0,3980]
L_max_y2 = ["Highest Price","Lowest Price"]

plt.barh(L_max_y2,H_max_x2)
plt.title("TCS Stock -  Maximum Highest and Lowest Price",size=25)
 
for index, value in enumerate(H_max_x2):
    plt.text(value, index,
             str(value))


# # High vs Low - Reliance

# In[9]:


plt.style.use("dark_background")
h_l_r=pd.DataFrame()
h_l_r["r_high"]=pd.Series(df1.High)
h_l_r["r_low"]=pd.Series(df1.Low)
h_l_r["Date"]=df1.Date
h_l_r=h_l_r.set_index(["Date"])
h_l_r.iplot(kind="bar")


# # High vs Low - TCS

# In[10]:


plt.style.use("dark_background")
h_l_tcs=pd.DataFrame()
h_l_tcs["tcs_high"]=pd.Series(df2.High)
h_l_tcs["tcs_low"]=pd.Series(df2.Low)
h_l_tcs["Date"]=df2.Date
h_l_tcs=h_l_tcs.set_index(["Date"])
h_l_tcs.iplot(kind="bar",color=["green","lightgreen"])


# # Preprocessing

# # 1. Reliance

# In[6]:


df_df1=df1.drop(["Symbol","Series","Prev Close","VWAP","Trades","Deliverablevolume","Deliverable"],axis=1)


# In[7]:


df_df1


# In[8]:


df1_close_sqrt=sqrt(df_df1["Close"])
df1_close_log=log(df_df1["Close"])


# In[14]:


fig,ax=plt.subplots(2,3)
sns.histplot(df1["Close"],ax=ax[0,0]) ; ax[0,0].set_title("Original")
#ax[1]=plt.plot(df1["Close"])
ax[1,0].plot(df_df1.Close)
sns.histplot(df1_close_sqrt,ax=ax[0,1]) ; ax[0,1].set_title("sqrt")
ax[1,1].plot(df1_close_sqrt)
sns.histplot(df1_close_log,ax=ax[0,2]) ; ax[0,2].set_title("log")
ax[1,2].plot(df1_close_log)
plt.rcParams["figure.figsize"]=(18,9)


# In[10]:


# dicky fuller test
adft=adfuller(df1_close_log)
output=pd.Series(adft[0:4],index=["t-value","p-value","no.of.lags","no.of.observations"])
output


# # 2. TCS

# In[11]:


df_df2=df2.drop(["Symbol","Series","Prev Close","VWAP","Trades","Deliverablevolume","Deliverable"],axis=1)
df_df2


# In[12]:


df2_close_sqrt=sqrt(df_df2["Close"])
df2_close_log=log(df_df2["Close"])


# In[13]:


fig,ax=plt.subplots(2,3)
sns.histplot(df2["Close"],ax=ax[0,0]) ; ax[0,0].set_title("Original")
#ax[1]=plt.plot(df1["Close"])
ax[1,0].plot(df_df2.Close)
plt.rcParams["figure.figsize"]=(18,9)
sns.histplot(df2_close_sqrt,ax=ax[0,1]) ; ax[0,1].set_title("sqrt")
ax[1,1].plot(df2_close_sqrt)
sns.histplot(df2_close_log,ax=ax[0,2]) ; ax[0,2].set_title("log")
ax[1,2].plot(df2_close_log)


# In[15]:


adft2=adfuller(df2_close_log)
output=pd.Series(adft2[0:4],index=["t-value","p-value","no.of.lags","no.of.observations"])
output


# # Model Building ( Reliance )

# In[16]:


train1=df1_close_log.iloc[0:int(len(df1_close_log)*.80)]
test1=df1_close_log.iloc[int(len(df1_close_log)*.80):int(len(df1_close_log))]
train1.shape,test1.shape


# In[17]:


# 1. Simple Exponential Smoothing
decimals=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

best_one=[]
for i in range(len(decimals)):
    model=SimpleExpSmoothing(train1).fit(smoothing_level=decimals[i])
    preds=model.predict(start=test1.index[0],end=test1.index[-1])
    best_one.append(mean_absolute_error(preds,test1))
for j in range(len(decimals)):
    print(decimals[j],":",best_one[j])


# In[18]:


# 1. Simple Exponential Smoothing
df1_model1=SimpleExpSmoothing(train1).fit(smoothing_level=1.0)
df1_preds1=df1_model1.predict(start=test1.index[0],end=test1.index[-1])
df1_error1=mean_absolute_error(df1_preds1,test1)
df1_error1


# In[19]:


# 2. Advance Exponential technique ( HOLT ) 
decimals=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
best_one=[]
for i in range(len(decimals)):
    model=Holt(train1).fit(smoothing_level=1.0,smoothing_trend=decimals[i])
    preds=model.predict(start=test1.index[0],end=test1.index[-1])
    best_one.append(mean_absolute_error(preds,test1))
    
for j in range(len(decimals)):
    print(decimals[j],":",best_one[j])


# In[20]:


# 2. Advance Exponential technique ( HOLT ) 
df1_model2=Holt(train1).fit(smoothing_level=1.0,smoothing_trend=0.1)
df1_preds2=df1_model2.predict(start=test1.index[0],end=test1.index[-1])
df1_error2=mean_absolute_error(df1_preds2,test1)
df1_error2


# In[21]:


# 3. Holt's winter additive trend and aditive sesonality
df1_model3=ExponentialSmoothing(train1,trend="add",seasonal="add",seasonal_periods=12).fit()
df1_preds3=df1_model3.predict(start=test1.index[0],end=test1.index[-1])
df1_error3=mean_absolute_error(df1_preds3,test1)
df1_error3


# In[22]:


# 4. Holt's winter additive trend and multiplicative sesonality
df1_model4=ExponentialSmoothing(train1,trend="add",seasonal="mul",seasonal_periods=12).fit()
df1_preds4=df1_model4.predict(start=test1.index[0],end=test1.index[-1])
df1_error4=mean_absolute_error(df1_preds4,test1)
df1_error4


# In[23]:


# 5. Auto Regressive
df1_model5 = AutoReg(train1, lags=10).fit()
print(df1_model5.summary())


# In[24]:


len(train1)


# In[25]:


df1_preds5= df1_model5.predict(start=len(train1), end=2474, dynamic=False)
df1_error5=mean_absolute_error(df1_preds5,test1)
df1_error5


# In[26]:


test1.shape


# In[27]:


# 6. ARIMA Method
df1_model6 = ARIMA(train1, order=(3,1,0))
df1_model6 = df1_model6.fit()
a,b,c=df1_model6.forecast(495)
d=pd.Series(a,index=test1.index)
df1_error6=mean_absolute_error(d,test1)
df1_error6


# In[28]:


train1.shape,test1.shape


# In[29]:


# 7. LSTM

#df1_lstm=pd.DataFrame()
#df1_lstm["1_day_back_price"]=df1_close_log.shift(1)
#df1_lstm["2_day_back_price"]=df1_close_log.shift(2)
#df1_lstm["3_day_back_price"]=df1_close_log.shift(3)
#df1_lstm["close"]=df1_close_log
#df1_lstm["Date"]=df_df1["Date"]
#df1_lstm=df1_lstm.set_index(["Date"])
#df1_lstm=df1_lstm.dropna()

#df1_lstm_x=df1_lstm.drop(["close"],axis=1)
#df1_lstm_y=df1_lstm["close"]

#df1_lstm_xtrain=df1_lstm_x[:1980]
#df1_lstm_xtest=df1_lstm_x[1980:]
#df1_lstm_ytrain=df1_lstm_y[:1980]
#df1_lstm_ytest=df1_lstm_y[1980:]

#n_features=1
#df1_lstm_xtrain=np.asarray(df1_lstm_xtrain)
#df1_lstm_xtrain=df1_lstm_xtrain.reshape(df1_lstm_xtrain.shape[0],df1_lstm_xtrain.shape[1],n_features)
#df1_lstm_xtest=np.asarray(df1_lstm_xtest)
#df1_lstm_xtest=df1_lstm_xtest.reshape(df1_lstm_xtest.shape[0],df1_lstm_xtest.shape[1],n_features)

#df1_lstm_ytrain=np.asarray(df1_lstm_ytrain)
#df1_lstm_ytest=np.asarray(df1_lstm_ytest)


normalizing1=MinMaxScaler(feature_range=(0,1))
df1_lstm=normalizing1.fit_transform(np.array(df1_close_log).reshape(-1,1))

train_data1=int(len(df1_lstm)*0.80)
test_data1=len(df1_lstm)-train_data1
train_data1,test_data1


# In[30]:


df1_train,df1_test=df1_lstm[0:train_data1,:],df1_lstm[train_data1:len(df1_lstm),:]


# In[31]:


df1_train.shape,df1_test.shape


# In[32]:


def create_df1(dataset,step):
    xxtrain,yytrain=[],[]
    for i in range(len(dataset)-step-1):
        a=dataset[i:(i+step),0]
        xxtrain.append(a)
        yytrain.append(dataset[i+step,0])
    return np.array(xxtrain),np.array(yytrain)


# In[33]:


t_s1=100
df1_lstm_xtrain,df1_lstm_ytrain=(create_df1(df1_train,t_s1))
df1_lstm_xtest,df1_lstm_ytest=(create_df1(df1_test,t_s1))


# In[34]:


df1_lstm_xtrain.shape,df1_lstm_ytrain.shape,df1_lstm_xtest.shape,df1_lstm_ytest.shape


# In[35]:


n_features=1
df1_lstm_xtrain=df1_lstm_xtrain.reshape(df1_lstm_xtrain.shape[0],df1_lstm_xtrain.shape[1],n_features)
df1_lstm_xtest=df1_lstm_xtest.reshape(df1_lstm_xtest.shape[0],df1_lstm_xtest.shape[1],n_features)


# In[36]:


df1_lstm_xtrain.shape,df1_lstm_xtest.shape


# In[37]:


df1_model7 = Sequential()
df1_model7.add(LSTM(50,return_sequences=True,input_shape=(df1_lstm_xtrain.shape[1],1)))
df1_model7.add(LSTM(50,return_sequences=True))
df1_model7.add(LSTM(50))
df1_model7.add(Dense(1,activation='linear')) 
df1_model7.compile(optimizer="adam",loss="MAE")
df1_model7.fit(df1_lstm_xtrain,df1_lstm_ytrain,validation_data=(df1_lstm_xtest,df1_lstm_ytest),epochs=10,batch_size=64)


# In[38]:


df1_lstm_ytest.shape


# In[39]:


fut_inp1=df1_lstm_ytest[294:]
fut_inp1=fut_inp1.reshape(1,-1)
temp_inp1=list(fut_inp1)
fut_inp1.shape


# In[40]:


temp_inp1=temp_inp1[0].tolist()


# In[41]:


lst_out1=[]
n_steps=100
i=0
while(i<30):
    
    if(len(temp_inp1)>100):
        fut_inp1=np.array(temp_inp1[1:])
        fut_inp1=fut_inp1.reshape(1,-1)
        fut_inp1=fut_inp1.reshape((1,n_steps,1))
        yhat=df1_model7.predict(fut_inp1,verbose=0)
        temp_inp1.extend(yhat[0].tolist())
        temp_inp1=temp_inp1[1:]
        lst_out1.extend(yhat.tolist())
        i=i+1
    else:
        fut_inp1=fut_inp1.reshape((1,n_steps,1))
        yhat=df1_model7.predict(fut_inp1,verbose=0)
        temp_inp1.extend(yhat[0].tolist())
        lst_out1.extend(yhat.tolist())
        i=i+1
               
    print(lst_out1) 


# In[42]:


# 30 days forcast - reliance
np.exp(normalizing1.inverse_transform(lst_out1))


# In[43]:


df1_preds7=df1_model7.predict(df1_lstm_xtest)


# In[44]:


df1_error7=mean_absolute_error(df1_preds7,df1_lstm_ytest)
df1_error7


# In[45]:


plt.plot(df1_preds7)
plt.plot(df1_lstm_ytest)


# In[46]:


plot_new1=np.arange(1,101)
plot_pred1=np.arange(101,131)


# In[47]:


df1_lstm.shape


# In[48]:


plt.plot(plot_new1,df1_lstm[2375:])
plt.plot(plot_pred1,lst_out1)


# In[49]:


2475-394


# In[50]:


df1_lstm_xtrain.shape,df1_lstm_ytest.shape,df1_lstm.shape


# In[51]:


ranges1=np.arange(2081,2475)
ranges2=np.arange(2475,2505)
c1=np.exp(normalizing1.inverse_transform(df1_lstm))
c2=np.exp(normalizing1.inverse_transform(df1_model7.predict(df1_lstm_xtrain)))
c3=np.exp(normalizing1.inverse_transform(df1_preds7))
c4=np.exp(normalizing1.inverse_transform(lst_out1))
plt.plot(c1)
plt.plot(c2)
plt.plot(ranges1,c3)
plt.plot(ranges2,c4)
label=["close","train","test","forcast"]
plt.legend(label)


# In[52]:


df1_models=dict()
df1_models["SES"]=1-df1_error1
df1_models["AES (HOLT'S)"]=1-df1_error2
df1_models["AES add(trend,sesonal)"]=1-df1_error3
df1_models["AES add(trend),mul(sesonal)"]=1-df1_error4
df1_models["AutoRegressive"]=1-df1_error5
df1_models["ARIMA"]=1-df1_error6
df1_models["LSTM"]=1-df1_error7


# In[53]:


df1_models=pd.DataFrame(list(df1_models.items()),columns=["Model","Accuracy"])


# In[54]:


df1_models=df1_models.sort_values(["Accuracy"],ascending=False)
df1_models


# In[55]:


sns.barplot(x="Accuracy",y="Model",data=df1_models)


# # Finalize Model for reliance is ( LSTM ) - supply entire data init

# In[68]:


df1_x,df1_y=create_df1(df1_lstm,t_s1)
df1_x.shape,df1_y.shape


# In[69]:


df1_x=df1_x.reshape(df1_x.shape[0],df1_x.shape[1],n_features)


# In[70]:


df1_model = Sequential()
df1_model.add(LSTM(50,return_sequences=True,input_shape=(df1_x.shape[1],1)))
df1_model.add(LSTM(50,return_sequences=True))
df1_model.add(LSTM(50))
df1_model.add(Dense(1,activation='linear')) 
df1_model.compile(optimizer="adam",loss="MAE")
df1_model.fit(df1_x,df1_y,epochs=10,batch_size=64)


# In[71]:


df1_y.shape


# In[72]:


fut_inp=df1_y[2274:]
fut_inp=fut_inp.reshape(1,-1)
temp_inp=list(fut_inp)
fut_inp.shape


# In[73]:


temp_inp=temp_inp[0].tolist()


# In[74]:


lst_out=[]
n_steps=100
i=0
while(i<30):
    
    if(len(temp_inp)>100):
        fut_inp=np.array(temp_inp[1:])
        fut_inp=fut_inp.reshape(1,-1)
        fut_inp=fut_inp.reshape((1,n_steps,1))
        yhat=df1_model.predict(fut_inp,verbose=0)
        temp_inp.extend(yhat[0].tolist())
        temp_inp=temp_inp[1:]
        lst_out.extend(yhat.tolist())
        i=i+1
    else:
        fut_inp=fut_inp.reshape((1,n_steps,1))
        yhat=df1_model.predict(fut_inp,verbose=0)
        temp_inp.extend(yhat[0].tolist())
        lst_out.extend(yhat.tolist())
        i=i+1
               
    print(lst_out) 


# In[75]:


np.exp(normalizing1.inverse_transform(lst_out))


# In[76]:


df1_entire_data=df1_lstm.tolist()
df1_entire_data.extend(lst_out)


# In[77]:


plt.plot(df1_entire_data)


# # Generating file for Deployment - Reliance

# In[80]:


df1_model.save("relaince_model.h5")


# #  TCS - models

# In[81]:


train2=df2_close_log.iloc[0:int(len(df2_close_log)*.80)]
test2=df2_close_log.iloc[int(len(df2_close_log)*.80):int(len(df2_close_log))]


# In[82]:


train2.shape,test2.shape


# In[83]:


# 1.Simple Exponential Smoothing
decimals=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
best_one=[]
for i in range(len(decimals)):
    model=SimpleExpSmoothing(train2).fit(smoothing_level=decimals[i])
    preds=model.predict(start=test2.index[0],end=test2.index[-1])
    best_one.append(mean_absolute_error(preds,test2))
for j in range(len(decimals)):
    print(decimals[j],":",best_one[j])


# In[84]:


# 2. Simple Exponential Smoothing
df2_model1=SimpleExpSmoothing(train2).fit(smoothing_level=1.0)
df2_preds1=df2_model1.predict(start=test2.index[0],end=test2.index[-1])
df2_error1=mean_absolute_error(df2_preds1,test2)
df2_error1


# In[85]:


# 2. Advance Exponential Smoothing ( HOLT )
decimals=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
best_one=[]
for i in range(len(decimals)):
    model=Holt(train2).fit(smoothing_level=1.0,smoothing_trend=decimals[i])
    preds=model.predict(start=test2.index[0],end=test2.index[-1])
    best_one.append(mean_absolute_error(preds,test2))
for j in range(len(decimals)):
    print(decimals[j],":",best_one[j])


# In[86]:


# 2. Advance Exponential Smoothing ( HOLT )
df2_model2=Holt(train2).fit(smoothing_level=1.0,smoothing_trend=0.1)
df2_preds2=df2_model2.predict(start=test2.index[0],end=test2.index[-1])
df2_error2=mean_absolute_error(df2_preds2,test2)
df2_error2


# In[87]:


# 3. Holt's winter additive trend and aditive sesonality
df2_model3=ExponentialSmoothing(train2,trend="add",seasonal="add",seasonal_periods=12).fit()
df2_preds3=df2_model3.predict(start=test2.index[0],end=test2.index[-1])
df2_error3=mean_absolute_error(df2_preds3,test2)
df2_error3


# In[88]:


# 4. Holt's winter additive trend and multiplicative sesonality
df2_model4=ExponentialSmoothing(train2,trend="add",seasonal="mul",seasonal_periods=12).fit()
df2_preds4=df2_model4.predict(start=test2.index[0],end=test2.index[-1])
df2_error4=mean_absolute_error(df2_preds4,test2)
df2_error4


# In[89]:


# 6. Auto Regressive
df2_model5 = AutoReg(train2, lags=10).fit()
print(df2_model5.summary())


# In[90]:


df2_preds5 = df2_model5.predict(start=len(train2), end=2474, dynamic=False)
df2_error5=mean_absolute_error(df2_preds5,test2)
df2_error5


# In[91]:


test2.shape


# In[92]:


# 6. ARIMA Method
df2_model6 = ARIMA(train2, order=(3,1,0))
df2_model6 = df2_model6.fit()
a,b,c=df2_model6.forecast(495)
d=pd.Series(a,index=test2.index)
df2_error6=mean_absolute_error(d,test2)
df2_error6


# In[93]:


# 8. LSTM

#df2_lstm=pd.DataFrame()
#df2_lstm["1_day_back_price"]=df2_close_log.shift(1)
#df2_lstm["2_day_back_price"]=df2_close_log.shift(2)
#df2_lstm["3_day_back_price"]=df2_close_log.shift(3)
#df2_lstm["close"]=df2_close_log
#df2_lstm["Date"]=df_df2["Date"]
#df2_lstm=df2_lstm.set_index(["Date"])
#df2_lstm=df2_lstm.dropna()
#df2_lstm_x=df2_lstm.drop(["close"],axis=1)
#df2_lstm_y=df2_lstm[["close"]]
#df2_lstm_close=np.array(df2_close_log).reshape(-1,1)

#df2_lstm_x=np.array(df2_lstm_x).reshape(-1,1)
#df2_lstm_y=np.array(df2_lstm_y).reshape(-1,1)

#df2_lstm_xtrain=df2_lstm_x[:1980]
#df2_lstm_xtest=df2_lstm_x[1980:]
#df2_lstm_ytrain=df2_lstm_y[:1980]
#df2_lstm_ytest=df2_lstm_y[1980:]

normalizing2=MinMaxScaler(feature_range=(0,1))
df2_lstm=normalizing2.fit_transform(np.array(df2_close_log).reshape(-1,1))


# In[94]:


train_data2=int(len(df2_lstm)*0.80)
test_data2=len(df2_lstm)-train_data2
train_data2,test_data2


# In[95]:


df2_lstm_train,df2_lstm_test=df2_lstm[0:train_data2,:],df2_lstm[train_data2:len(df2_lstm),:]


# In[96]:


len(df2_lstm_train),len(df2_lstm_test)


# In[97]:


def create_df2(dataset,step):
    xxtrain,yytrain=[],[]
    for i in range(len(dataset)-step-1):
        a=dataset[i:(i+step),0]
        xxtrain.append(a)
        yytrain.append(dataset[i+step,0])
    return np.array(xxtrain),np.array(yytrain)    


# In[98]:


t_s2=100
df2_lstm_xtrain,df2_lstm_ytrain=(create_df2(df2_lstm_train,t_s2))
df2_lstm_xtest,df2_lstm_ytest=(create_df2(df2_lstm_test,t_s2))


# In[99]:


df2_lstm_xtrain.shape,df2_lstm_ytrain.shape,df2_lstm_xtest.shape,df2_lstm_ytest.shape


# In[100]:


df2_lstm_xtrain=df2_lstm_xtrain.reshape(df2_lstm_xtrain.shape[0],df2_lstm_xtrain.shape[1],n_features)
df2_lstm_xtest=df2_lstm_xtest.reshape(df2_lstm_xtest.shape[0],df2_lstm_xtest.shape[1],n_features)


# In[101]:


df2_model7 = Sequential()
df2_model7.add(LSTM(50,return_sequences=True,input_shape=(df2_lstm_xtrain.shape[1],1)))
df2_model7.add(LSTM(50,return_sequences=True))
df2_model7.add(LSTM(50))
df2_model7.add(Dense(1,activation='linear')) 
df2_model7.compile(optimizer="adam",loss="MAE")
df2_model7.fit(df2_lstm_xtrain,df2_lstm_ytrain,validation_data=(df2_lstm_xtest,df2_lstm_ytest),epochs=10,batch_size=64)


# In[102]:


df2_preds7=df2_model7.predict(df2_lstm_xtest)
df2_error7=mean_absolute_error(df2_preds7,df2_lstm_ytest)
df2_error7


# In[103]:


plt.plot(df2_preds7)
plt.plot(df2_lstm_ytest)


# In[104]:


df2_lstm_ytest.shape


# In[105]:


fut_inp2=df2_lstm_ytest[294:]
fut_inp2=fut_inp2.reshape(1,-1)


# In[106]:


temp_inp2=list(fut_inp2)
fut_inp2.shape


# In[107]:


temp_inp2=temp_inp2[0].tolist()


# In[108]:


#n=int(input("Enter the number"))
lst_out2=[]
n_steps=100
i=0
while(i<30):
    
    if(len(temp_inp2)>100):
        fut_inp2=np.array(temp_inp2[1:])
        fut_inp2=fut_inp2.reshape(1,-1)
        fut_inp2=fut_inp2.reshape((1,n_steps,1))
        yhat=df2_model7.predict(fut_inp2,verbose=0)
        temp_inp2.extend(yhat[0].tolist())
        temp_inp2=temp_inp2[1:]
        lst_out2.extend(yhat.tolist())
        i=i+1
    else:
        fut_inp2=fut_inp2.reshape((1,n_steps,1))
        yhat=df2_model7.predict(fut_inp2,verbose=0)
        temp_inp2.extend(yhat[0].tolist())
        lst_out2.extend(yhat.tolist())
        i=i+1
               
    print(lst_out2) 


# In[109]:


np.exp(normalizing2.inverse_transform(lst_out2))


# In[110]:


plot_new2=np.arange(1,101)
plot_pred2=np.arange(101,131)


# In[111]:


df2_lstm.shape,len(lst_out2)


# In[112]:


plt.plot(plot_new2,normalizing2.inverse_transform(np.exp(df2_lstm[2375:])))
plt.plot(plot_pred2,normalizing2.inverse_transform(np.exp(lst_out2)))


# In[113]:


ranges1=np.arange(2081,2475)
ranges2=np.arange(2475,2505)
t1=np.exp(normalizing2.inverse_transform(df2_lstm))
t2=np.exp(normalizing2.inverse_transform(df2_model7.predict(df2_lstm_xtrain)))
t3=np.exp(normalizing2.inverse_transform(df2_preds7))
t4=np.exp(normalizing2.inverse_transform(lst_out2))
plt.plot(t1)
plt.plot(t2)
plt.plot(ranges1,t3)
plt.plot(ranges2,t4)
label=["close","train","test","forcast"]
plt.legend(label)


# In[114]:


df2_models=dict()
df2_models["SES"]=1-df2_error1
df2_models["AES (HOLT'S)"]=1-df2_error2
df2_models["AES add(trend,sesonal)"]=1-df2_error3
df2_models["AES add(trend),mul(sesonal)"]=1-df2_error4
df2_models["AutoRegressive"]=1-df2_error5
df2_models["ARIMA"]=1-df2_error6
df2_models["LSTM"]=1-df2_error7


# In[115]:


df2_models=pd.DataFrame(list(df2_models.items()),columns=["Model","Accuracy"])
df2_models


# In[116]:


df2_models=df2_models.sort_values("Accuracy",ascending=False)
df2_models


# In[117]:


sns.barplot(x="Accuracy",y="Model",data=df2_models)


# # Finalize Model for TCS is ( LSTM ) - supply entire data init

# In[118]:


df2_x,df2_y=create_df2(df2_lstm,t_s2)
df2_x.shape,df2_y.shape


# In[119]:


df2_x=df2_x.reshape(df2_x.shape[0],df2_x.shape[1],n_features)


# In[120]:


df2_model = Sequential()
df2_model.add(LSTM(50,return_sequences=True,input_shape=(df2_x.shape[1],1)))
df2_model.add(LSTM(50,return_sequences=True))
df2_model.add(LSTM(50))
df2_model.add(Dense(1,activation='linear')) 
df2_model.compile(optimizer="adam",loss="MAE")
df2_model.fit(df2_x,df2_y,epochs=10,batch_size=64)


# In[122]:


df2_y.shape


# In[130]:


fut_inp_=df2_y[2274:]
fut_inp_=fut_inp_.reshape(1,-1)
temp_inp_=list(fut_inp_)
fut_inp_.shape


# In[131]:


temp_inp_=temp_inp_[0].tolist()


# In[132]:


lst_out_=[]
n_steps=100
i=0
while(i<30):
    
    if(len(temp_inp_)>100):
        fut_inp_=np.array(temp_inp_[1:])
        fut_inp_=fut_inp_.reshape(1,-1)
        fut_inp_=fut_inp_.reshape((1,n_steps,1))
        yhat=df2_model.predict(fut_inp_,verbose=0)
        temp_inp_.extend(yhat[0].tolist())
        temp_inp_=temp_inp_[1:]
        lst_out_.extend(yhat.tolist())
        i=i+1
    else:
        fut_inp_=fut_inp_.reshape((1,n_steps,1))
        yhat=df2_model.predict(fut_inp_,verbose=0)
        temp_inp_.extend(yhat[0].tolist())
        lst_out_.extend(yhat.tolist())
        i=i+1
               
    print(lst_out_) 


# In[133]:


np.exp(normalizing2.inverse_transform(lst_out_))


# In[134]:


df2_entire_data=df2_lstm.tolist()
df2_entire_data.extend(lst_out_)


# In[137]:


len(df2_entire_data)


# In[138]:


plt.plot(df2_entire_data)


# # For Checking Purpose 

# In[139]:


df2_model.save("tcs_model.h5")


# In[142]:


import tensorflow


# In[143]:


model = tensorflow.keras.models.load_model("tcs_model.h5")


# In[145]:


p=model.predict(df2_lstm_xtest)


# In[148]:


np.exp(normalizing2.inverse_transform(p))

