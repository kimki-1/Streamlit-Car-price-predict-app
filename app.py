import streamlit as st 
import tensorflow as tf
import numpy as np
import pandas as pd 
import tensorflow.keras

from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def main() :
    st.title('자동차 가격 예측')
    menu = [ 'Home', '자동차 가격 예측', 'About' ]
    choice = st.sidebar.selectbox('메뉴', menu)

    df = pd.read_csv('data/Car_Purchasing_Data.csv') 

    X = df.iloc[: , 3 :-2+1 ]
    y = df['Car Purchase Amount']

    sc_X = MinMaxScaler()
    X_scaled = sc_X.fit_transform(X)

    y = y.values.reshape( -1, 1)
    sc_y = MinMaxScaler()
    y_scaled = sc_y.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size =0.25, random_state = 50)

    if choice == 'Home':

        st.dataframe(df)

        if  st.button('데이터가공'):
            st.markdown('#### X 데이터')
            st.dataframe(X)
            st.markdown('#### y 데이터')
            st.dataframe(y)

        if st.button('Scalerd') :
            st.markdown('#### X Scaled')
            st.write(X_scaled)      
            st.markdown('#### y Scaled')
            st.write(y_scaled)

        if st.button('테스트셋, 트레이닝셋 분류') :
            st.markdown('#### X train')
            st.write(X_train)
            st.markdown('#### X test')
            st.write(X_test)
            st.markdown('#### y train')
            st.write(y_train)
            st.markdown('#### y test')
            st.write(y_test)
    
    elif choice == '자동차 가격 예측':

        gender = st.radio('성별 선택 ', ['여자', '남자'])
        if gender == '여자' :
            gender = 0 
        elif gender == '남자' :
            gender = 1

        age = st.number_input('나이', 1, 120)
        salary = st.number_input('연봉($)',0)
        card = st.number_input('카드 빛($)',0)
        worth = st.number_input('순자산($)',0)

        model = tf.keras.models.load_model('data/car_ai.h5')
        new_data = np.array( [ gender, age, salary, card, worth ])
        new_data = new_data.reshape(1, -1)
        new_data_scaled = sc_X.transform(new_data)

        y_pred = model.predict(new_data_scaled)
        y_pred_orginal = sc_y.inverse_transform(y_pred)

        if st.button('예측') :
            st.title('당신의 자동차 가격은 {}$ 입니다'.format(y_pred_orginal[0][0]))

if __name__ == '__main__' :
    main()