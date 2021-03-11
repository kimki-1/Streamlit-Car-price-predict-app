import streamlit as st 
import tensorflow as tf
import numpy as np
import pandas as pd 
import tensorflow.keras
import matplotlib.pyplot as plt 
import matplotlib
import seaborn as sns
matplotlib.use('Agg')

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
    
    columns = car_df.columns
    columns = list(columns)
    
    selected_columns = st.multiselect('컬럼을 선택하시오', columns)
    
    if len(selected_columns) != 0 :
        st.dataframe(car_df[selected_columns])
    else :
        st.write('선택한 컬럼이 없습니다.')
    
    # 상관계수를 화면에 보여주도록 만듭니다. 
    # 멀티셀렉트에 컬럼명을 보여주고,
    # 해당 컬럼들에 대한 상관 계수를 보여주세요.
    # 단, 컬럼들은 숫자 컬럼들만 멀티셀렉트에 나타나야합니다.
    
    
    corr_columns = car_df.columns[ car_df.dtypes != object ] # car_df.corr().columns.values
    selected_corr = st.multiselect('상관계수 컬럼을 선택', corr_columns)

    if len(selected_corr) != 0 :
        st.dataframe(car_df[selected_corr].corr())
        # 위에서 선택한 컬럼들을 이용해서, 시본의 페어플롯을 그린다
        fig = sns.pairplot(data = car_df[selected_corr])
        st.pyplot(fig)

    else :
        st.write('선택한 컬럼이 없습니다.')

    # 컬럼을 하나만 선택하면, 해당 컬럼의 max,min에 해당하는
    # 사람의 데이터를 화면에 보여주는 기능
    float_columns = car_df.columns[ car_df.dtypes == float ] 
    selected_col = st.selectbox('컬럼 선택', float_columns )

    min_data = car_df.loc[car_df[selected_col].min() == car_df[selected_col],]
    st.markdown('### Min Data')
    st.dataframe(min_data)

    max_data = car_df.loc[car_df[selected_col].max() == car_df[selected_col],]
    st.markdown('### Max Data')
    st.dataframe(max_data)

    # 고객이름을 검색 할 수 있는 기능 
    fname = st.text_input('이름을 검색하세요')
    find_customer = car_df['Customer Name'].str.contains(fname, case =False)
    if len(fname) != 0  :
        st.dataframe(car_df.loc[find_customer, ] )
    else :
        st.write('이름을 입력하세요')

