import streamlit as st 
import tensorflow as tf
import numpy as np
import pandas as pd 
import tensorflow.keras

from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def run_eda_app() :
    st.subheader('EDA 화면 입니다.')

    car_df = pd.read_csv('data/Car_Purchasing_Data.csv', encoding='ISO-8859-1') 

    radio_menu = ['데이터 프레임', '통계치']
    selected_radio = st.radio('선택하세요', radio_menu)
    
    if selected_radio == '데이터 프레임' :
        st.dataframe(car_df)

    elif selected_radio == '통계치' :
        st.dataframe(car_df.describe())



