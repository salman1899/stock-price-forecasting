# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import tensorflow
from streamlit_option_menu import option_menu
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")



model1=tensorflow.keras.models.load_model("reliance_model.h5")

model2=tensorflow.keras.models.load_model("tcs_model.h5")


def create_df1(dataset,step):
    xxtrain,yytrain=[],[]
    for i in range(len(dataset)-step-1):
        a=dataset[i:(i+step),0]
        xxtrain.append(a)
        yytrain.append(dataset[i+step,0])
    return np.array(xxtrain),np.array(yytrain)



options = option_menu("Main Menu",["Home"], 
icons=['house','gear-fill',"envelope"], menu_icon="cast", default_index=0,orientation="horizontal")
st.title("STOCK MARKET FORECASTING")
   
if options == "Home":
    
  a=st.sidebar.selectbox("STOCKS",("Select the stock","Reliance","TCS"))
    
  if a == "Reliance": 
   # Reading Dataset   
   df1=pd.read_csv("reliance_data.csv")
   st.dataframe(df1)
   
   # Plotting Close Price
   st.subheader("closing price")
   fig=plt.figure(figsize=(12,6))
   plt.plot(df1.Close)
   st.pyplot(fig)
   
   # Plotting Close Price with MA100
   st.subheader("closing price with 100MA")
   ma1_100=df1.Close.rolling(100).mean()
   plt.plot(ma1_100)
   st.pyplot(fig)
   
   # Perform using Close Price   
   df1_=pd.read_csv("reliance_data.csv",index_col="Date")
   df1_=df1_["Close"] 
   
   # Performing LOG & SCALING 
   df1_log=np.log(df1_)
   normalizing1=MinMaxScaler(feature_range=(0,1))
   df1_norm=normalizing1.fit_transform(np.array(df1_log).reshape(-1,1))
                                         
   t_s1=100                            
   df1_x,df1_y=create_df1(df1_norm, t_s1)                                
   fut_inp1=df1_y[2274:]
   fut_inp1=fut_inp1.reshape(1,-1)   
   temp_inp1=list(fut_inp1) 
   temp_inp1=temp_inp1[0].tolist()
      
   lst_out1=[]   
   n_steps=100
   i=0   
   
   st.subheader("How many days you want to forecast")
   #int_val = st.slider('Seconds', min_value=0, max_value=366, value=1, step=1)
   int_val = st.slider('Days', min_value=1, max_value=366, step=1)

   while(i<int_val):
    if(len(temp_inp1)>100):
        fut_inp1=np.array(temp_inp1[1:])
        fut_inp1=fut_inp1.reshape(1,-1)
        fut_inp1=fut_inp1.reshape((1,n_steps,1))
        yhat=model1.predict(fut_inp1,verbose=0)
        temp_inp1.extend(yhat[0].tolist())
        temp_inp1=temp_inp1[1:]
        lst_out1.extend(yhat.tolist())
        i=i+1
    else:
        fut_inp1=fut_inp1.reshape((1,n_steps,1))
        yhat=model1.predict(fut_inp1,verbose=0)
        temp_inp1.extend(yhat[0].tolist())
        lst_out1.extend(yhat.tolist())
        i=i+1

   lst_out1=normalizing1.inverse_transform(lst_out1)
   lst_out1=np.exp(lst_out1) 
   st.dataframe(lst_out1)
   
   c1=lst_out1
   fig3_=plt.figure(figsize=(12,6))
   plt.plot(c1)
   st.pyplot(fig3_)
   
   
# TCS PART
  
  elif a == "TCS":
      
   # Reading Dataset 
   df2=pd.read_csv("tcs_data.csv")
   st.dataframe(df2)
   
   # Plotting Close Price
   st.subheader("closing price")
   fig3=plt.figure(figsize=(12,6))
   plt.plot(df2.Close)
   st.pyplot(fig3)
   
   # Plotting Close Price with MA100
   st.subheader("closing price with 100MA")
   ma1_100=df2.Close.rolling(100).mean()
   plt.plot(ma1_100)
   st.pyplot(fig3)
   
   df2_=pd.read_csv("tcs_data.csv")
   df2_=df2_["Close"] 
   
   # Performing LOG & SCALING 
   df2_log=np.log(df2_)
   normalizing2=MinMaxScaler(feature_range=(0,1))
   df2_norm=normalizing2.fit_transform(np.array(df2_log).reshape(-1,1))
                                         
   t_s2=100                            
   df2_x,df2_y=create_df1(df2_norm, t_s2)                                
   fut_inp2=df2_y[2274:]
   fut_inp2=fut_inp2.reshape(1,-1)   
   temp_inp2=list(fut_inp2) 
   temp_inp2=temp_inp2[0].tolist()
      
   lst_out2=[]   
   n_steps=100
   i=0   
   
   st.subheader("How many days you want to forecast")
   int_val_ = st.slider('Days', min_value=1, max_value=366, step=1)
   #int_val = st.number_input('Seconds', min_value=1, max_value=10, value=5, step=1)
   while(i<int_val_):
    if(len(temp_inp2)>100):
        fut_inp2=np.array(temp_inp2[1:])
        fut_inp2=fut_inp2.reshape(1,-1)
        fut_inp2=fut_inp2.reshape((1,n_steps,1))
        yhat=model2.predict(fut_inp2,verbose=0)
        temp_inp2.extend(yhat[0].tolist())
        temp_inp2=temp_inp2[1:]
        lst_out2.extend(yhat.tolist())
        i=i+1
    else:
        fut_inp2=fut_inp2.reshape((1,n_steps,1))
        yhat=model2.predict(fut_inp2,verbose=0)
        temp_inp2.extend(yhat[0].tolist())
        lst_out2.extend(yhat.tolist())
        i=i+1
        
            
   lst_out2=normalizing2.inverse_transform(lst_out2)
   lst_out2=np.exp(lst_out2) 
   st.dataframe(lst_out2)
   
   fig4=plt.figure(figsize=(12,6))
   plt.plot(lst_out2)
   st.pyplot(fig4)
      
       






