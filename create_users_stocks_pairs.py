import pandas as pd
import numpy as np
def create_users_stocks_pairs(classified_user_portfolio, X_investment_norm_df, Scaled_User_PortFolio):

    # Extracting the weights and normalized investment values
    weights = classified_user_portfolio[['Capital Preservation', 'Liquidity & Accessibilty', 'Modest Growth', 
                                     'Stable Income Generation', 'Potential Growth', 'Moderate Volatility']].values
    X_investment_norm_values = X_investment_norm_df[['CPS', 'ALS', 'MGS', 'SIGS', 'PGS', 'VS']].values.T

    # Performing dot product to get the weighted sum
    ss_values = np.dot(weights, X_investment_norm_values)

    # Extracting necessary columns once
    portfolio_values = classified_user_portfolio['Portfolio'].values
    scaled_portfolio_values = Scaled_User_PortFolio.values
    investment_values = X_investment_norm_df.values

    # Creating a comprehensive list of columns for the final DataFrame
    columns = ['User ID', 'Portfolio', 'Budget in Usd', 'Time frame in Months', 'Risk Tolerance %', 'Capital Preservation',
               'Liquidity & Accessibilty', 'Modest Growth', 'Stable Income Generation', 'Potential Growth', 'Moderate Volatility',
               'Sector & Industry Focus', 'Stock ID', 'Open', 'Close', 'High','Low' ,'Volume', 'Shares Out Standing', 'Float Shares',
               'Shares Percent SharesOut', 'Dividends', 'Dividend Yield', 'Payout Ratio', 'Stock Splits', 'Revenue Growth',
               'Profit Margins', 'Market Capital', 'Free Cash Flow', 'Debt To Equity', 'beta', 'Overall Risk', 'return on equity',
               'Earnings per share', 'ForwardEps', 'pegRatio', 'Forward PE', 'StrongBuy', 'Buy', 'Hold', 'Sell', 'StrongSell',
               'RecommendationMean', 'Sector_Basic Materials', 'Sector_Communication Services', 'Sector_Consumer Cyclical',
               'Sector_Consumer Defensive', 'Sector_Energy', 'Sector_Financial Services', 'Sector_Healthcare', 'Sector_Industrials',
               'Sector_Real Estate', 'Sector_Technology', 'Sector_Utilities', 'CPS', 'ALS', 'MGS', 'SIGS', 'PGS', 'VS', 'SS']

    # List to collect rows
    all_data_rows = []

    # Populating DataFrame rows
    for i in range(len(scaled_portfolio_values)):
        user_row = scaled_portfolio_values[i]
        budget = user_row[2]
        base_values = list(user_row)  # All columns from Scaled_PortFolio_df
        base_values.insert(1, portfolio_values[i])  # Insert portfolio value at the correct position

        for j in range(len(investment_values)):
            stock_row = investment_values[j]
            ss = ss_values[i][j] + 12 * budget
            row = base_values + list(stock_row) + [ss]
            all_data_rows.append(row)

    # Creating DataFrame all at once
    users_stocks_pairs = pd.DataFrame(all_data_rows, columns=columns)

    
    return users_stocks_pairs