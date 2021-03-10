import streamlit as st 
import tensorflow as tf
import numpy as np
import pandas as pd 
import tensorflow.keras

from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib 

def run_ml_app():
    st.subheader('Machine Learning')

    model = tf.keras.models.load_model('data/car_ai.h5')
    sc_X = joblib.load('data/sc_X.pkl')
    sc_y = joblib.load('data/sc_y.pkl')

    gender = st.radio('성별 선택 ', ['여자', '남자'])
    if gender == '여자' :
        gender = 0 
    elif gender == '남자' :
        gender = 1

    age = st.number_input('나이', 1, 120)
    salary = st.number_input('연봉($)',0)
    card = st.number_input('카드 빛($)',0)
    worth = st.number_input('순자산($)',0)
    
    new_data = np.array( [ gender, age, salary, card, worth ])
    new_data = new_data.reshape(1, -1)
    new_data_scaled = sc_X.transform(new_data)

    y_pred = model.predict(new_data_scaled)
    y_pred_orginal = sc_y.inverse_transform(y_pred)

    if st.button('예측') :
        st.title('당신의 자동차 가격은 {}$ 입니다'.format(y_pred_orginal[0][0]))
