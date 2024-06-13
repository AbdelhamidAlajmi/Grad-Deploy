# بسم اللّه الرحمن الرحيم و به نستعين
#الحمد للّه رب العالمين، والصلاة والسلام على أشرف الأنبياء والمرسلين، نبيّنا محمد وعلى آله وصحبه أجمعين، ومن تبعهم بإحسان إلى يوم الدين

# Importing Libs

import yfinance as yf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import time
import concurrent.futures
import sklearn
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from classify_portfolio import classify_portfolio
from create_users_stocks_pairs import create_users_stocks_pairs


# Define Finuction To Fetch Data From YFinance Lib

def fetch_data(symbol):
    comp = yf.Ticker(symbol)
    hist_data = comp.history(period="3m")  # Retrieve historical data
    if not hist_data.empty: # Check if data is available
        company_name = comp.info.get('shortName')
        last_row = hist_data.iloc[-1]  # Get the last row (latest data)
        last_date = hist_data.index[-1]  # Get the date of the last row
        open_price = last_row['Open']
        close_price = last_row['Close']
        high_price = last_row['High']
        volume = last_row['Volume']
        dividends = last_row['Dividends']
        stock_splits = last_row['Stock Splits']
        shares = comp.info.get('sharesOutstanding')  # Retrieve shares outstanding
        market_capital = comp.info.get('marketCap')  # Retrieve market capitalization
        sector = comp.info.get('sector')
        Revenue_Growth=comp.info.get('revenueGrowth')
        dividend_yield=comp.info.get('dividendYield')
        overall_risk=comp.info.get('overallRisk')
        pay_out_Ratio=comp.info.get('payoutRatio') 
        beta=comp.info.get('beta') #Capital Preservastion #Lower
        profit_Margins=comp.info.get('profitMargins')
        forward_Eps=comp.info.get('forwardEps') 
        pegRatio=comp.info.get('pegRatio')
        floatShares=comp.info.get('floatShares')
        sharesPercentSharesOut=comp.info.get('sharesPercentSharesOut')
        debtToEquity=comp.info.get('debtToEquity') #Capital Preservastion ##Lower
        forwardPE=comp.info.get('forwardPE') #Capital Preservastion ##Lower
        sharesOutstanding=comp.info.get('sharesOutstanding')
        FreeCashFlow=comp.info.get('freeCashflow') 
        StrongBuy=comp.recommendations.iloc[0,1]
        Buy=comp.recommendations.iloc[0,2]
        Hold=comp.recommendations.iloc[0,3]
        Sell=comp.recommendations.iloc[0,4]
        StrongSell=comp.recommendations.iloc[0,5]
        recommendationMean = comp.info.get('recommendationMean')
        # Earnings per share
        k=comp.quarterly_financials
        l = k.T
        if 'Basic EPS' in l.columns:
            EarningsPerShare = l['Basic EPS'].iloc[0]
        else:
            EarningsPerShare = None  # or any other value you want to assign in case the column doesn't exist
         # Return on equity
        returnOnEquity = comp.info.get('returnOnEquity')
            
        return {'Date':last_date,'Symbol': symbol, 'Company': company_name,
                'Open': open_price,'Close': close_price, 'High': high_price,
                'Volume': volume,'Shares Out Standing':sharesOutstanding,'Float Shares':floatShares,'Shares Percent SharesOut':sharesPercentSharesOut,
                'Dividends': dividends, 'Dividend Yield':dividend_yield,'Payout Ratio':pay_out_Ratio,
                'Stock Splits':stock_splits,'Revenue Growth':Revenue_Growth,
                'Profit Margins':profit_Margins,
                'Market Capital':market_capital,'Free Cash Flow':FreeCashFlow,'Debt To Equity':debtToEquity,'beta':beta,'Overall Risk':overall_risk,
                'return on equity':returnOnEquity,
                'Earnings per share':EarningsPerShare,'ForwardEps':forward_Eps,
                'pegRatio':pegRatio,'Forward PE':forwardPE,
                'StrongBuy':StrongBuy,'Buy':Buy,'Hold':Hold,'Sell':Sell,'StrongSell':StrongSell,'RecommendationMean':recommendationMean,
                'Sector': sector}

# Symbols of Stocks To Fetch

symbols = ['MSFT','NVDA','AAPL','AMZN','META','GOOGL','GOOG','LLY','AVGO','JPM','XOM','TSLA','UNH','V','PG','COST','MA','JNJ','MRK','HD','ABBV',
           'WMT','NFLX','BAC','CVX','AMD','KO','CRM','QCOM','PEP','TMO','LIN','WFC','ADBE','ORCL','AMAT','DIS','ABT','CSCO','MCD','ACN','TXN','GE',
           'DHR','VZ','CAT','AMGN','PM','INTU','PFE','NEE','IBM','CMCSA','MU','GS','ISRG','NOW','RTX','UBER','UNP','HON','SPGI','COP','AXP','BKNG',
           'LRCX','INTC','ETN','T','ELV','LOW','VRTX','PGR','TJX','MS','SYK','C','NKE','ADI','BSX','MDT','SCHW','BA','CB','KLAC','BLK','REGN','MMC',
           'PLD','ADP','LMT','UPS','CI','PANW','DE','TMUS','SBUX','AMT','MDLZ','FI','SNPS','BMY','BX','SO','CMG','MO','ZTS','GILD','CDNS','APH','DUK',
           'MCK','CL','ICE','CVS','ANET','TT','WM','TDG','PYPL','FCX','CME','EQIX','NXPI','EOG','BDX','SHW','CEG','TGT','HCA','PH','GD','ITW','CSX',
           'ABNB','MPC','SLB','MCO','APD','MSI','EMR','NOC','PNC','ECL','USB','PSX','ROP','CTAS','FDX','ORLY','AON','MAR','WELL','PCAR','MMM','AJG',
           'GM','COF','AIG','VLO','CARR','EW','HLT','MCHP','NSC','WMB','SPG','MRNA','ROST','TRV','F','JCI','DLR','AZO','TFC','NEM','SRE','OKE','CPRT',
           'ADSK','AEP','AFL','TEL','BK','FIS','KMB','GEV','DXCM','O','PSA','URI','CCI','MET','D','AMP','HUM','ALL','DHI','PRU','IDXX','LHX','HES',
           'STZ','OXY','AME','OTIS','SMCI','GWW','IQV','PWR','PCG','DOW','PAYX','COR','A','YUM','NUE','LEN','RSG','MSCI','FTNT','KMI','VRSK','GIS',
           'MNST','ACGL','CNC','MPWR','CMI','IR','RCL','LULU','PEG','CTVA','FAST','EXC','SYY','KDP','KVUE','FANG','DD','MLM','IT','KR','CTSH','EA',
           'XYL','ADM','VMC','HWM','BIIB','BKR','DAL','FICO','GEHC','ED','EXR','ON','DFS','HPQ','MTD','CSGP','RMD','HAL','ODFL','HIG','XEL','DVN',
           'PPG','CDW','FSLR','VST','TSCO','WAB','ROK','VICI','HSY','EFX','EIX','GLW','ANSS','DG','AVB','EL','EBAY','CHTR','DECK','TRGP','KHC','HPE',
           'CHD','WTW','CBRE','FTV','TROW','NTAP','IRM','TTWO','GPN','WEC','GRMN','AWK','DOV','IFF','WDC','LYB','FITB','CAH','PHM','NVR','KEYS','MTB',
           'WST','ZBH','DTE','BR','DLTR','ETR','EQR','NDAQ','RJF','STT','APTV','STE','VLTO','TER','BALL','WY','BRO','CTRA','PTC','SBAC','PPL','ES',
           'INVH','AXON','FE','VTR','GPC','TYL','HUBB','LDOS','CNP','STX','COO','STLD','ULTA','CPAY','AEE','EXPD','DPZ','TDY','ALGN','SYF','AVY','BLDR',
           'CBOE','CINF','NRG','HBAN','WBD','MOH','WAT','ENPH','OMC','ARE','DRI','CMS','LUV','J','UAL','PFG','HOLX','ILMN','ESS','ATO','NTRS','MKC','TXT',
           'EQT','RF','BBY','BAX','CCL','LH','MRO','LVS','PKG','EG','CLX','EXPE','WRB','MAA','CFG','VRSN','TSN','DGX','K','ZBRA','IP','FDS','BG','IEX',
           'CF','JBL','SWKS','MAS','CE','AMCR','SNA','CAG','GEN','L','TRMB','AES','AKAM','DOC','RVTY','PODD','ALB','POOL','JBHT','ROL','PNR','WRK','KEY',
           'HST','LYV','LNT','VTRS','SWK','KIM','LW','EMN','EVRG','TECH','NDSN','UDR','JKHY','IPG','SJM','UHS','NI','CPT','WBA','JNPR','LKQ','MGM','INCY',
           'CRL','KMX','BBWI','NWSA','ALLE','CTLT','EPAM','TPR','AOS','REG','QRVO','CHRW','FFIV','HII','TFX','TAP','MOS','AIZ','APA','WYNN','HRL','HSIC',
           'MTCH','GNRC','PNW','CPB','BXP','FOXA','BWA','ETSY','SOLV','DAY','CZR','DVA','AAL','RL','HAS','MKTX','FRT','NCLH','PAYC','GL','FMC','IVZ','RHI',
           'BEN','CMA','MHK','BIO','PARA']

# Fetch Data From YFinance Lib

investment_data = []
with concurrent.futures.ThreadPoolExecutor() as executor:
    results = executor.map(fetch_data, symbols)
    for result in results:
        if result:
            investment_data.append(result)

investment = pd.DataFrame(investment_data)

# One-hot encode the 'Sector' column
investment = pd.concat([investment, pd.get_dummies(investment['Sector'], prefix='Sector',dtype=int)], axis=1)

# Drop the original 'Sector' column
investment.drop(columns=['Sector'], inplace=True)


# Drop Nulls

investment.dropna(inplace=True,ignore_index=True)

# Create X & Normalize it

scaler = MinMaxScaler()
X_investment_norm= scaler.fit_transform(investment.drop(['Date','Symbol','Company'],axis=1))
X_investment_norm_df=pd.DataFrame(X_investment_norm,columns=['Open', 'Close', 'High', 'Volume',
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
portfolio_df=pd.read_csv('portfolio.CSV')

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
