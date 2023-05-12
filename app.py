#importing important libraries
import numpy as np
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import date
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import datetime as datetime
#setting the favicon and title of page
st.set_page_config(page_title="Stock Nerd", page_icon=":chart_with_upwards_trend:", layout="wide")

#remove default
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
# Sidebar
#title
st.sidebar.title('STOCK NERD')
st.sidebar.subheader("ONE STOP STOCK ANALYSIS WEB APP")
st.sidebar.write('---')
#query paremeters
st.sidebar.subheader('Query parameters')
start_date = st.sidebar.date_input("Start date", datetime.date(2013, 1, 1))
end_date = st.sidebar.date_input("End date", datetime.date(2023, 3, 31))
if end_date < start_date:#validation of date
    st.error("Error: End date should be greater than or equal to start date.")
    st.stop()
if end_date < start_date:
    st.error("Error: End date should be greater than or equal to start date.")
    st.stop()
yf.pdr_override()
tickerlist= pd.read_csv('list.csv') 
tickerSymbol = st.sidebar.selectbox(
    'Choose the Stock Ticker',tickerlist)
#ticker main page
df= yf.download(tickerSymbol,start_date,end_date)
if df.empty:
    st.error("Error: No data available for the selected ticker and date range.")
    st.stop()
tickerData=yf.Ticker(tickerSymbol)
#dashbroad info
longname = tickerData.info["longName"]
st.header('**%s**' % longname)
if not longname:
    st.header(tickerSymbol)    
curency = tickerData.info['financialCurrency']
st.markdown('**%s**' % curency)
if not longname:
    st.markdown("NA")   
pe= tickerData.info["forwardPE"]
pe=int(pe)
if not pe:
    st.markdown("NA")   
avg200= tickerData.info["twoHundredDayAverage"]
avg200=int(avg200)
if not avg200:
    st.markdown("NA")   
mc= tickerData.info["marketCap"]
mc=int(mc)
if not mc:
    st.markdown("NA")   
wh= tickerData.info["fiftyTwoWeekHigh"]
if not wh:
    st.markdown("NA")   
wl=tickerData.info["fiftyTwoWeekLow"]
if not wl:
    st.markdown("NA")   

with st.container():
    col1, col2, col3,col4 = st.columns(4)
    col1.metric("PE RATIO", pe, "")
    col2.metric("200AVG", avg200, "")
    col3.metric("52WeekHigh",wh,"")
    col4.metric("52WeekLow",wl,"")
#tabbed pane
tab1, tab2, tab3,tab4,tab5= st.tabs(["Raw Data","Summarized Data","Close price vs Day ", "Moving Averages","Original vs Prediction"])
#describing data 
with tab1:
    st.subheader('RAW DATA')
    st.table(df.tail(7))
    #download data
    @ st.cache_data
    def convert_df(df):
        return df.to_csv().encode('utf-8')
    csv = convert_df(df)
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name=tickerSymbol,
        mime='text/csv',
    )
    
#summary data
with tab2:
    st.subheader('SUMMARIZED DATA')
    st.table(df.describe())
with tab3:
#CLOSING PRICE CHART
    st.subheader('CLOSING PRICE VS DAY CHART')
    chart_data = pd.DataFrame({'date': df.index, 'price': df.Close})
    chart_data = chart_data.melt(id_vars=['date'], value_vars=['price'], var_name='series', value_name='value')
    brush = alt.selection(type='interval', encodings=['x'])

    chart = alt.Chart(chart_data).mark_line().encode(
        x='date:T',
        y='value:Q',
        color='series:N',
        tooltip=['date:T', 'value:Q']
    ).add_selection(
        brush
    ).properties(
        width=1000,
        height=400
    )
    st.altair_chart(chart)
with tab4:
    #Moving averages
    ma25t,ma50t,ma100t,ma200t = st.tabs(["25MA","50MA","100MA", "200Ma"])
    #25 days moving averages
    with ma25t:
        st.subheader('25 DAYS MOVING AVERAGE')
        ma25 = df.Close.rolling(25).mean()
        chart_data = pd.DataFrame({'date': df.index, 'price': df.Close, 'MA25': ma25})
        chart_data = chart_data.melt(id_vars=['date'], value_vars=['price', 'MA25'], var_name='series', value_name='value')

        brush = alt.selection(type='interval', encodings=['x'])

        chart = alt.Chart(chart_data).mark_line(color='red').encode(
        x='date:T',
        y='value:Q',
        color='series:N',
        tooltip=['date:T', 'value:Q']
        ).add_selection(
        brush
        ).properties(
        width=1000,
        height=400
    )
        st.altair_chart(chart)
 #50 days moving averages
    with ma50t:
         st.subheader('50 DAYS MOVING AVERAGE')
         ma50 = df.Close.rolling(50).mean()
         chart_data = pd.DataFrame({'date': df.index, 'price': df.Close,'MA25': ma25 ,'MA50': ma50})
         chart_data = chart_data.melt(id_vars=['date'], value_vars=['price', 'MA50','MA25'], var_name='series', value_name='value')
         brush = alt.selection(type='interval', encodings=['x'])

         chart = alt.Chart(chart_data).mark_line(color='red').encode(
            x='date:T',
            y='value:Q',
            color='series:N',
            tooltip=['date:T', 'value:Q']
        ).add_selection(
        brush
        ).properties(
        width=1000,
        height=400
        )
         st.altair_chart(chart)
    with ma100t:
    #100 DAYS MOVING AVERAGE PLOTTING
        st.subheader('100 Days Moving Average')
        ma100 = df.Close.rolling(100).mean()
        chart_data = pd.DataFrame({'date': df.index, 'price': df.Close,'MA50': ma50, 'MA100': ma100})
        chart_data = chart_data.melt(id_vars=['date'], value_vars=['price','MA50', 'MA100'], var_name='series', value_name='value')

        brush = alt.selection(type='interval', encodings=['x'])

        chart = alt.Chart(chart_data).mark_line().encode(
        x='date:T',
        y='value:Q',
        color='series:N',
        tooltip=['date:T', 'value:Q']
    ).add_selection(
        brush
    ).properties(
        width=1000,
        height=400
        )
        st.altair_chart(chart) 
    with ma200t:
    #PLOTTING 200 DAYS MOVING AVERAGES WITH 100 DAYS MOVING AVERAGE
        st.subheader('200 Days Moving Average')
        ma200 = df.Close.rolling(200).mean()
        chart_data = pd.DataFrame({'date': df.index, 'price': df.Close, 'MA100': ma100,'MA200': ma200})
        chart_data = chart_data.melt(id_vars=['date'], value_vars=['price', 'MA100','MA200'], var_name='series', value_name='value')
        brush = alt.selection(type='interval', encodings=['x'])
        chart = alt.Chart(chart_data).mark_line().encode(
        x='date:T',
        y='value:Q',
        color='series:N',
        tooltip=['date:T', 'value:Q']
        ).add_selection(
        brush
        ).properties(
        width=1000,
        height=400
        )

        st.altair_chart(chart)
#TRAINING THE MODEL
#SPLITING DATA INTO TRAINING AND TESTING DATAFRAME
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing=pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])
#scaling the data
scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)
x_train = []
y_train = []
for i in range(100,data_training_array.shape[0]):
    x_train.append(data_training_array[i-100: i])
    y_train.append(data_training_array[i,0])   
#spliting data into x_train and y_train
x_train , y_train = np.array(x_train) , np.array(y_train)
#loading pretrained model
model =load_model('kera_model.h5')
#testing part
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing,ignore_index=True)
input_data = scaler.fit_transform(final_df)
x_test=[]
y_test=[]
for i in range(100 , input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i,0])   
x_test,y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler=scaler.scale_
scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor
#plotting original vs predicted
with tab5:
    st.subheader("original price vs predicted price")
    fig2=plt.figure(figsize=(20,10))
    plt.plot(y_test ,'b',label='original Price')
    plt.plot(y_predicted ,'r',label='predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig2)



#footer
with st.container():
    st.write("© 2023 StockNerd")

