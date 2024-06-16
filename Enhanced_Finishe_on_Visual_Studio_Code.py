# بسم اللّه الرحمن الرحيم و به نستعين
#الحمد للّه رب العالمين، والصلاة والسلام على أشرف الأنبياء والمرسلين، نبيّنا محمد وعلى آله وصحبه أجمعين، ومن تبعهم بإحسان إلى يوم الدين

# Importing Libs

import yfinance as yf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import concurrent.futures
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from classify_portfolio import classify_portfolio
from create_users_stocks_pairs import create_users_stocks_pairs
from Fetch_data import symbols
from Fetch_data import fetch


# Fetch Data From YFinance Lib for speeding in local

# print("concurrent start")

investment_data = []
with concurrent.futures.ThreadPoolExecutor() as executor:
    results = executor.map(fetch, symbols)
    for result in results:
        if result:
            investment_data.append(result)

# print("concurrent end")




# Fetch Data From YFinance Lib for no_concurrency in server
# investment_data = []

# # Sequentially fetch data for each symbol
# for symbol in symbols:
#     result = fetch(symbol)
#     if result:
#         investment_data.append(result)


investment_data = pd.DataFrame(investment_data)

# One-hot encode the 'Sector' column
investment = pd.concat([investment_data, pd.get_dummies(investment_data['Sector'], prefix='Sector',dtype=int)], axis=1)

# Drop the original 'Sector' column
investment.drop(columns=['Sector'], inplace=True)


# Drop Nulls

investment.dropna(inplace=True,ignore_index=True)

# Create X & Normalize it

scaler = MinMaxScaler()
X_investment_norm= scaler.fit_transform(investment.drop(['Date','Symbol','Company'],axis=1))
X_investment_norm_df=pd.DataFrame(X_investment_norm,columns=['Open', 'Close', 'High','Low' ,'Volume',
       'Shares Out Standing', 'Float Shares', 'Shares Percent SharesOut',
       'Dividends', 'Dividend Yield', 'Payout Ratio',
       'Stock Splits','Revenue Growth', 'Profit Margins',
       'Market Capital', 'Free Cash Flow', 'Debt To Equity', 'beta',
       'Overall Risk', 'return on equity', 'Earnings per share',
       'ForwardEps', 'pegRatio', 'Forward PE',
       'StrongBuy', 'Buy', 'Hold', 'Sell', 'StrongSell', 'RecommendationMean',
       'Sector_Basic Materials', 'Sector_Communication Services',
       'Sector_Consumer Cyclical', 'Sector_Consumer Defensive',
       'Sector_Energy', 'Sector_Financial Services', 'Sector_Healthcare',
       'Sector_Industrials', 'Sector_Real Estate', 'Sector_Technology',
       'Sector_Utilities'])



# Create New Rows & Calc Scores

X_investment_norm_df['CPS']=0
X_investment_norm_df['ALS']=0
X_investment_norm_df['MGS']=0
X_investment_norm_df['SIGS']=0
X_investment_norm_df['PGS']=0
X_investment_norm_df['VS']=0
for i in range(len(X_investment_norm_df)):
    X_investment_norm_df['CPS'][i]=0.20*X_investment_norm_df['Dividend Yield'][i]+0.20*(1-X_investment_norm_df['beta'][i])+0.20*(1-X_investment_norm_df['Forward PE'][i])+0.20*X_investment_norm_df['return on equity'][i]+0.20*(1-X_investment_norm_df['Debt To Equity'][i])
    X_investment_norm_df['ALS'][i]=0.5*X_investment_norm_df['Float Shares'][i]+0.5*X_investment_norm_df['Shares Out Standing'][i]
    X_investment_norm_df['MGS'][i]=0.4*X_investment_norm_df['Earnings per share'][i]+0.3*X_investment_norm_df['return on equity'][i]+0.3*(1-X_investment_norm_df['Forward PE'][i])
    X_investment_norm_df['SIGS'][i]=0.25*X_investment_norm_df['Dividend Yield'][i]+0.25*X_investment_norm_df['Payout Ratio'][i]+0.25*(1-X_investment_norm_df['Debt To Equity'][i])+0.25*X_investment_norm_df['Free Cash Flow'][i]
    X_investment_norm_df['PGS'][i]=0.16*X_investment_norm_df['Forward PE'][i]+0.16*X_investment_norm_df['return on equity'][i]+0.17*X_investment_norm_df['Earnings per share'][i]+0.17*X_investment_norm_df['Revenue Growth'][i]+0.17*X_investment_norm_df['pegRatio'][i]+0.17*X_investment_norm_df['Profit Margins'][i]
    X_investment_norm_df['VS'][i]=0.5*(1-X_investment_norm_df['beta'][i])+0.5*(1-X_investment_norm_df['Overall Risk'][i])

# Insert New Coulmn For Stock ID

X_investment_norm_df.insert(0,'Stock ID',X_investment_norm_df.index,False)

# User Data

# Read Portfolio dataset
portfolio_df=pd.read_csv('portfolio.csv')

# Create an empty DataFrame
random_user = pd.DataFrame()
# Generate random values for 'Budget in Usd', 'Time frame in Months', and 'Risk Tolerance %' columns
np.random.seed(1)
random_user['Budget in Usd'] = np.random.uniform(45000, 2837500, size=270)#Generate Random values for budget from 45000$ to 2837500$ with size = 10000
random_user['Time frame in Months'] = np.random.uniform(36, 120, size=270)#Generate Random Values for duration from 36 month to 120 month with size=10000  
random_user['Risk Tolerance %'] = np.random.uniform(10, 90, size=270)#Genertate Random Values For Risk Tolerance From 10 to 90 percentace with size =10000


classified_user_portfolio = classify_portfolio(random_user, portfolio_df)

# Normalize User Data

Scaled_User_PortFolio=scaler.fit_transform(classified_user_portfolio.drop(['Portfolio'],axis=1))

Scaled_User_PortFolio=pd.DataFrame(Scaled_User_PortFolio,columns=[['Budget in Usd', 'Time frame in Months', 'Risk Tolerance %',
       'Capital Preservation', 'Liquidity & Accessibilty','Modest Growth', 'Stable Income Generation', 'Potential Growth',
       'Moderate Volatility', 'Sector & Industry Focus']])

Scaled_User_PortFolio.insert(0,'User ID',Scaled_User_PortFolio.index,False)

# Create Y That Represents Simalarity Score

Users_Stocks_Pairs=create_users_stocks_pairs(classified_user_portfolio, X_investment_norm_df, Scaled_User_PortFolio)

# Prediction Model

stocks = Users_Stocks_Pairs.drop(columns=['User ID','Portfolio', 'Budget in Usd','Time frame in Months','Risk Tolerance %','Capital Preservation',
                                'Liquidity & Accessibilty','Modest Growth','Stable Income Generation','Potential Growth','Moderate Volatility',
                                'Sector & Industry Focus', 'Stock ID','SS'])
users = Users_Stocks_Pairs[['Budget in Usd','Time frame in Months','Risk Tolerance %','Capital Preservation',
                                'Liquidity & Accessibilty','Modest Growth','Stable Income Generation','Potential Growth','Moderate Volatility',
                                'Sector & Industry Focus']]
y=Users_Stocks_Pairs['SS']

user_train,user_test =train_test_split(users, train_size=0.80, shuffle=True, random_state=1)
stocks_train,stocks_test =train_test_split(stocks, train_size=0.80, shuffle=True, random_state=1)
y_train, y_test =train_test_split(y,train_size=0.80, shuffle=True, random_state=1)

num_user_features=user_train.shape[1]
num_stock_features=stocks_train.shape[1]

num_outputs = 10
user_NN = tf.keras.models.Sequential([
    tf.keras.layers.Dense(700,activation='relu'),
    tf.keras.layers.Dense(350,activation='relu'),
    tf.keras.layers.Dense(num_outputs)  
])

stock_NN = tf.keras.models.Sequential([  
    tf.keras.layers.Dense(700,activation='relu'),
    tf.keras.layers.Dense(350,activation='relu'),
    tf.keras.layers.Dense(num_outputs)  
])

input_user = tf.keras.layers.Input(shape=(num_user_features))
vu = user_NN(input_user)

# create the item input and point to the base network
input_stock = tf.keras.layers.Input(shape=(num_stock_features))
vs = stock_NN(input_stock)


# compute the dot product of the two vectors vu and vm
output = tf.keras.layers.Dot(axes=1)([vu, vs])

# specify the inputs and output of the model
model = tf.keras.Model([input_user, input_stock], output)

#model.summary()

cost_fn = tf.keras.losses.MeanSquaredError()
opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt,loss=cost_fn)

history=model.fit([user_train,stocks_train],y_train,epochs=7)

model.evaluate([user_test, stocks_test], y_test)


model.save('Final_Model.h5')
