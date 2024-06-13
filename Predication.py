import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from Enhanced_Finishe_on_Visual_Studio_Code import investment
from Preprocess import new_user_df

model_path=r'C:\Users\AJM\Grad\Api_Final\Final_Model.h5'
model=load_model(model_path)


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
    for i in range(num_of_recommended_stocks):
        recommended_companies.append({
            f"company_name_{i+1}": names_of_recommended_stocks[i],
            f"symbol_{i+1}": symbols_of_recommended_stocks[i],
            f"price_{i+1}": prices_of_recommended_stocks[i]
        })

    return recommended_companies