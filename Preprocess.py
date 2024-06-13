from create_users_stocks_pairs import create_users_stocks_pairs
from classify_portfolio import classify_portfolio
from Enhanced_Finishe_on_Visual_Studio_Code import classified_user_portfolio , portfolio_df , scaler , X_investment_norm_df 
import pandas as pd


new_user_df=pd.DataFrame(columns=classified_user_portfolio.columns)


def preprocess(new_user_data,classified_user_portfolio,create_users_stocks_pairs):


    new_user_id=new_user_data["user_id"]
    new_user_budget=new_user_data["user_budget"]
    new_user_time_frame=new_user_data["user_time"]
    new_user_risk_tolerance=new_user_data["user_risk"]
    
    new_user_df.loc[0] = [new_user_budget, new_user_time_frame, new_user_risk_tolerance, None, None, None, None, None, None, None, None]

    classified_new_user_df=classify_portfolio(new_user_df, portfolio_df)

    scaled_classified_new_user=scaler.transform(classified_new_user_df.drop(['Portfolio'],axis=1))

    scaled_classified_new_user_df=pd.DataFrame(scaled_classified_new_user,columns=[['Budget in Usd', 'Time frame in Months', 'Risk Tolerance %',
        'Capital Preservation', 'Liquidity & Accessibilty',
        'Modest Growth', 'Stable Income Generation', 'Potential Growth',
        'Moderate Volatility', 'Sector & Industry Focus']])

    scaled_classified_new_user_df.insert(0,'User ID',new_user_id,allow_duplicates=True)

    new_user_stocks_pairs=create_users_stocks_pairs(classified_new_user_df, X_investment_norm_df, scaled_classified_new_user_df)

    user_vecs =pd.DataFrame(new_user_stocks_pairs[['Budget in Usd','Time frame in Months','Risk Tolerance %','Capital Preservation',
                                'Liquidity & Accessibilty','Modest Growth','Stable Income Generation','Potential Growth','Moderate Volatility',
                                'Sector & Industry Focus']]).to_numpy()

    stocks_vecs = pd.DataFrame(new_user_stocks_pairs.drop(columns=['Budget in Usd','Time frame in Months','Risk Tolerance %','Capital Preservation',
                                'Liquidity & Accessibilty','Modest Growth','Stable Income Generation','Potential Growth','Moderate Volatility',
                                'Sector & Industry Focus', 'Stock ID','SS','User ID','Portfolio'])).to_numpy()



    return user_vecs , stocks_vecs