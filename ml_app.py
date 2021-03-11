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

    # 1. 입력받기
    gender = st.radio('성별 선택 ', ['여자', '남자'])
    if gender == '여자' :
        gender = 0 
    elif gender == '남자' :
        gender = 1

    age = st.number_input('나이', 0, 120)
    salary = st.number_input('연봉($)', 0)
    card = st.number_input('카드 빛($)', 0)
    worth = st.number_input('순자산($)', 0)
    
    # 2. 예측하기 
    # 2-1. 모델 불러오기
    model = tf.keras.models.load_model('data/car_ai.h5')

    # 2-2. 넘파이 어레이 만들기
    new_data = np.array( [ gender, age, salary, card, worth ])

    # 2-3. 피처 스케일링 
    new_data = new_data.reshape(1, -1)
    sc_X = joblib.load('data/sc_X.pkl')
    new_data_scaled = sc_X.transform(new_data)

    # 예측 결과는, 스케일링 된 결과이므로, 다시 돌려놔야 한다
    # 2-4. 예측한다 
    y_pred = model.predict(new_data_scaled)
    sc_y = joblib.load('data/sc_y.pkl')
    y_pred_orginal = sc_y.inverse_transform(y_pred)

    # 3. 결과를 화면에 보여준다
    if st.button('예측') :
        st.markdown('## 예측 결과 입니다.')
        st.markdown('### 당신은 {:,.1f}$ 금액의 차를 구매 할 수 있습니다.'.format(y_pred_orginal[0][0]))
    
