import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from Enhanced_Finishe_on_Visual_Studio_Code import investment , investment_data
from Preprocess import new_user_df


#for Local Testing
#model_path=r"C:\Users\AJM\Grad\Api_Final_Production\Final_Model.h5"
#model=load_model(model_path)

#for server
model=load_model(Final_Model.h5)


def prediction(preprocessed_data_user,preprocessed_data_stock,new_user_df):

    prediction = model.predict([preprocessed_data_user,preprocessed_data_stock])

    sorted_index=np.argsort(-prediction,axis=0).reshape(-1).tolist()

    sorted_SS=prediction[sorted_index]
    sorted_stocks_by_ss=investment.iloc[sorted_index]
    num_of_recommended_stocks=0
    if 10 <= new_user_df['Risk Tolerance %'][0] <30:
        num_of_recommended_stocks= 20
    elif 30 <= new_user_df['Risk Tolerance %'][0] <60:
        num_of_recommended_stocks= 15
    elif 60 <= new_user_df['Risk Tolerance %'][0]  <= 90:
        num_of_recommended_stocks= 10

    recommended_stocks=sorted_stocks_by_ss[:num_of_recommended_stocks]

    names_of_recommended_stocks=recommended_stocks['Company'].values.tolist()
    symbols_of_recommended_stocks=recommended_stocks['Symbol'].values.tolist()
    similarity_score_of_recommended_stocks=sorted_SS[:num_of_recommended_stocks].tolist()
    prices_of_recommended_stocks=recommended_stocks['Close'].values.tolist()


    recommended_companies = []
    for i,symbol in enumerate(symbols_of_recommended_stocks):
        recommended_companies.append({
            f"company_name_{i+1}": names_of_recommended_stocks[i],
            f"symbol_{i+1}": symbols_of_recommended_stocks[i],
            f"Open_price_{i+1}": investment_data.loc[investment_data['Symbol'] == symbol]['Open'].iloc[0],
            f"High_price_{i+1}": investment_data.loc[investment_data['Symbol'] == symbol]['High'].iloc[0],
            f"Low_price_{i+1}": investment_data.loc[investment_data['Symbol'] == symbol]['Low'].iloc[0],
            f"Close_price_{i+1}": investment_data.loc[investment_data['Symbol'] == symbol]['Close'].iloc[0],
            f"Volume_{i+1}": investment_data.loc[investment_data['Symbol'] == symbol]['Volume'].iloc[0],
            f"Market_capital_{i+1}": float(investment_data.loc[investment_data['Symbol'] == symbol]['Market Capital'].iloc[0]),
            f"sector_{i+1}": investment_data.loc[investment_data['Symbol'] == symbol]['Sector'].iloc[0],
            f"forward_eps_{i+1}": investment_data.loc[investment_data['Symbol'] == symbol]['ForwardEps'].iloc[0],
            f"revenue_growth_{i+1}": investment_data.loc[investment_data['Symbol'] == symbol]['Revenue Growth'].iloc[0],
            f"profit_margin_{i+1}": investment_data.loc[investment_data['Symbol'] == symbol]['Profit Margins'].iloc[0],
            f"dividend_yield_{i+1}": investment_data.loc[investment_data['Symbol'] == symbol]['Dividend Yield'].iloc[0],
            f"payout_ratio_{i+1}": investment_data.loc[investment_data['Symbol'] == symbol]['Payout Ratio'].iloc[0],
            f"StrongBuy_{i+1}": float(investment_data.loc[investment_data['Symbol'] == symbol]['StrongBuy'].iloc[0]),
            f"Buy_{i+1}": float(investment_data.loc[investment_data['Symbol'] == symbol]['Buy'].iloc[0]),
            f"Hold_{i+1}": float(investment_data.loc[investment_data['Symbol'] == symbol]['Hold'].iloc[0]),
            f"Sell_{i+1}": float(investment_data.loc[investment_data['Symbol'] == symbol]['Sell'].iloc[0]),
            f"StrongSell_{i+1}": float(investment_data.loc[investment_data['Symbol'] == symbol]['StrongSell'].iloc[0]),
            f"RecommendationMean_{i+1}": float(investment_data.loc[investment_data['Symbol'] == symbol]['RecommendationMean'].iloc[0])

        })

    return recommended_companies , symbols_of_recommended_stocks
