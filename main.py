from flask import Flask,request,jsonify
from Preprocess import preprocess
from Predication import prediction
from create_users_stocks_pairs import create_users_stocks_pairs
from Enhanced_Finishe_on_Visual_Studio_Code import classified_user_portfolio , investment_data
from Preprocess import new_user_df



app=Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    
    # Get the JSON data from the request
    input_data = request.get_json()

    # Apply preprocessing
    preprocessed_data_user, preprocessed_data_stock = preprocess(input_data, classified_user_portfolio, create_users_stocks_pairs)

    # Make prediction
    prediction_data , shared_symbols=prediction(preprocessed_data_user , preprocessed_data_stock , new_user_df )


    return jsonify({"recommended_Stocks":prediction_data})

    

   
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
