import pandas as pd
def classify_portfolio(user_df, portfolio_df):
    for i in range(len(user_df)):
        budget = user_df.iloc[i, 0]
        time_frame = user_df.iloc[i, 1]
        risk_tolerance = user_df.iloc[i, 2]

        # Define the conditions and corresponding portfolio index
        if budget < 45000 and time_frame < 36 and 10 <= risk_tolerance < 30:
            portfolio_index = 0
        elif budget < 45000 and 36 <= time_frame < 84 and 10 <= risk_tolerance < 30:
            portfolio_index = 1
        elif budget < 45000 and 84 <= time_frame <= 120 and 10 <= risk_tolerance < 30:
            portfolio_index = 2
        elif 45000 <= budget < 1350000 and time_frame < 36 and 10 <= risk_tolerance < 30:
            portfolio_index = 3
        elif 45000 <= budget < 1350000 and 36 <= time_frame < 84 and 10 <= risk_tolerance < 30:
            portfolio_index = 4
        elif 45000 <= budget < 1350000 and 84 <= time_frame <= 120 and 10 <= risk_tolerance < 30:
            portfolio_index = 5
        elif 1350000 <= budget <= 2837500 and time_frame < 36 and 10 <= risk_tolerance < 30:
            portfolio_index = 6
        elif 1350000 <= budget <= 2837500 and 36 <= time_frame < 84 and 10 <= risk_tolerance < 30:
            portfolio_index = 7
        elif 1350000 <= budget <= 2837500 and 84 <= time_frame <= 120 and 10 <= risk_tolerance < 30:
            portfolio_index = 8
        elif budget < 45000 and time_frame < 36 and 30 <= risk_tolerance < 60:
            portfolio_index = 9
        elif budget < 45000 and 36 <= time_frame < 84 and 30 <= risk_tolerance < 60:
            portfolio_index = 10
        elif budget < 45000 and 84 <= time_frame <= 120 and 30 <= risk_tolerance < 60:
            portfolio_index = 11
        elif 45000 <= budget < 1350000 and time_frame < 36 and 30 <= risk_tolerance < 60:
            portfolio_index = 12
        elif 45000 <= budget < 1350000 and 36 <= time_frame < 84 and 30 <= risk_tolerance < 60:
            portfolio_index = 13
        elif 45000 <= budget < 1350000 and 84 <= time_frame <= 120 and 30 <= risk_tolerance < 60:
            portfolio_index = 14
        elif 1350000 <= budget <= 2837500 and time_frame < 36 and 30 <= risk_tolerance < 60:
            portfolio_index = 15
        elif 1350000 <= budget <= 2837500 and 36 <= time_frame < 84 and 30 <= risk_tolerance < 60:
            portfolio_index = 16
        elif 1350000 <= budget <= 2837500 and 84 <= time_frame <= 120 and 30 <= risk_tolerance < 60:
            portfolio_index = 17
        elif budget < 45000 and time_frame < 36 and 60 <= risk_tolerance <= 90:
            portfolio_index = 18
        elif budget < 45000 and 36 <= time_frame < 84 and 60 <= risk_tolerance <= 90:
            portfolio_index = 19
        elif budget < 45000 and 84 <= time_frame <= 120 and 60 <= risk_tolerance <= 90:
            portfolio_index = 20
        elif 45000 <= budget < 1350000 and time_frame < 36 and 60 <= risk_tolerance <= 90:
            portfolio_index = 21
        elif 45000 <= budget < 1350000 and 36 <= time_frame < 84 and 60 <= risk_tolerance <= 90:
            portfolio_index = 22
        elif 45000 <= budget < 1350000 and 84 <= time_frame <= 120 and 60 <= risk_tolerance <= 90:
            portfolio_index = 23
        elif 1350000 <= budget <= 2837500 and time_frame < 36 and 60 <= risk_tolerance <= 90:
            portfolio_index = 24
        elif 1350000 <= budget <= 2837500 and 36 <= time_frame < 84 and 60 <= risk_tolerance <= 90:
            portfolio_index = 25
        elif 1350000 <= budget <= 2837500 and 84 <= time_frame <= 120 and 60 <= risk_tolerance <= 90:
            portfolio_index = 26
        else:
            continue
        
        # Update the random_user_portfolio with the selected portfolio values
        for column in portfolio_df.columns:
            user_df.loc[i, column] = portfolio_df.loc[portfolio_index, column]
    
    return user_df