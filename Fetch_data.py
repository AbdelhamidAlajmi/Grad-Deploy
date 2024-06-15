import yfinance as yf
import pandas as pd

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




# Define Finuction To Fetch Data From YFinance Lib

def fetch(symbol):
    comp = yf.Ticker(symbol)
    hist_data = comp.history(period="1y")  # Retrieve historical data
    if not hist_data.empty: # Check if data is available
        company_name = comp.info.get('shortName')
        last_row = hist_data.iloc[-1]  # Get the last row (latest data)
        last_date = hist_data.index[-1]  # Get the date of the last row
        open_price = last_row['Open']
        close_price = last_row['Close']
        high_price = last_row['High']
        low_price=last_row['Low']
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
                'Open': open_price,'Close': close_price, 'High': high_price,'Low': low_price,
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

