# -*- coding: utf-8 -*-
#Importing the necessary libraries.
import bluetooth
import numpy as np
import pandas as pd
import math
import time
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error,max_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import RPi.GPIO as GPIO
#Be ready for the bluetooth connection.
print("Connection waiting...")
server_sock=bluetooth.BluetoothSocket(bluetooth.RFCOMM)
port=bluetooth.PORT_ANY
server_sock.bind(("",port))
server_sock.listen(1)
client_sock,adress=server_sock.accept()
print("Connection established with: ",adress)
print("Client: ",client_sock)
#Importing the dataset for ML.
dataset=pd.read_csv('dataset.csv')
x_s21=dataset.iloc[:,:2]
x_s31=dataset.iloc[:,:2]
y_s21=dataset.iloc[:,3]
y_s31=dataset.iloc[:,4]
#Feature scaling.
x_s21_train,x_s21_test,y_s21_train,y_s21_test=train_test_split(x_s21,y_s21,test_size=0.33,random_state=1)
x_s31_train,x_s31_test,y_s31_train,y_s31_test=train_test_split(x_s31,y_s31,test_size=0.33,random_state=1)
#Adding the Random Forrest Regressor for predicting S21 value.
rf_s21=RandomForestRegressor(n_estimators=10,random_state=0)
rf_s21.fit(x_s21_train,y_s21_train)
y_s21_pred=rf_s21.predict(x_s21_test)
#Adding the Random Forrest Regressor for predicting S31 value.
rf_s31=RandomForestRegressor(n_estimators=10,random_state=0)
rf_s31.fit(x_s31_train,y_s31_train)
y_s31_pred=rf_s31.predict(x_s31_test)
#Necessary settings for using PWM.
GPIO.setmode(GPIO.BCM)
GPIO.setup(19,GPIO.OUT)
DC_PWM=GPIO.PWM(19,1000)
DC_PWM.start(0)
#Ready for prediction.
print("Ready for prediction...")
#####
while True:
    data=client_sock.recv(1024)
    data=data.decode("utf-8")
    print("Data comes: ",data)
    #Preparing the incoming data for using in the prediction part.
    if(data!="exit"):
        #The data will split from comma to list.The first element of the list is freq and the second is voltage.
        freq,volt=data.split(",")
        freq=float(freq)
        volt=float(volt)
        print("Frequency: ",freq)
        print("Voltage: ",volt)
        lst=[volt,freq]
        #TODO 
        #The PWM part will be added.
        #The data will prepared for using in the predict function.
        wannapredict=np.reshape(lst,(1,2))
        pred_res_s21=rf_s21.predict(wannapredict)
        pred_res_s31=rf_s31.predict(wannapredict)
        print("The predicted S21 value is equal to: ",pred_res_s21)
        print("The predicted S31 value is equal to: ",pred_res_s31)
        total_pred=pred_s21+","+pred_s31
        print(total_pred)
        client_sock.send(total_pred)
    if(volt>15):
        volt=15
        client_sock.send("The voltage value must be lower then 15 V.")
    if(data=="exit"):
        client_sock.send("Connection closed.")
        print("Exit.")
        time.sleep(1)
        break
DC_PWM.stop()
GPIO.cleanup()
client_socket.close()
server_socket.close()
        





        


