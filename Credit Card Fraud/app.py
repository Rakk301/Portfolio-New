import gradio as gradio
import joblib as joblib
import pip
# pip install gradio
# pip install joblib
# pip install xgboost
# pip install scikit-learn

import joblib
import numpy as np
import gradio as gr

# Load Model and Scaler:
xgboost_model = joblib.load('xgboost_model_new.pkl')
scaler = joblib.load('scaler.pkl')

month_to_number = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}

def time_of_dayy(hour):
    if 6 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 18:
        return 'Afternoon'
    elif 18 <= hour < 24:
        return 'Evening'
    else:
        return 'Night'


options = [
    'Food/Dining',
    'Gas/Transport',
    'Online Grocery',
    'In-Person Grocery',
    'Miscellaneous Online',
    'Miscellaneous In-Person',
    'Shopping Online',
    'Shopping In-Person',
    'Personal Care',
    'Health/Fitness',
    'Kids/Pets',
    'Home',
    'Travel',
    'Entertainment'
]

category_options = [
    'Food/Dining',
    'Gas/Transport',
    'Online Grocery',
    'In-Person Grocery',
    'Miscellaneous Online',
    'Miscellaneous In-Person',
    'Shopping Online',
    'Shopping In-Person',
    'Personal Care',
    'Health/Fitness',
    'Kids/Pets',
    'Home',
    'Travel'
]

def predict_credit_card_fraud(amount, city_pop, month, hour, age, gender, category):

    month = month_to_number[month]
    time_of_day = time_of_dayy(hour)
    input_data = np.array([[amount, city_pop, month, hour, age, int(gender == 'M'),
                            int(time_of_day == 'Night'), int(time_of_day == 'Evening'), int(time_of_day == 'Morning')] +
                           [int(category == cat) for cat in category_options]])

    input_data[:, 0:2] = scaler.transform(input_data[:, 0:2])
    probability = xgboost_model.predict_proba(input_data)[:, 1]
    prediction = xgboost_model.predict(input_data)

    label = "Fraudulent" if prediction[0] == 1 else "Non-Fraudulent"
    return round(probability[0], 2), label

gender_options = ["M", "F"]
months = list(month_to_number.keys())

iface = gr.Interface(fn=predict_credit_card_fraud, 
                    inputs=[
                        gr.Number(label="Amount", info="Enter the Amount of the Transaction in Dollars"),
                        gr.Number(label="City Population", info="Enter the City Population"),
                        gr.Dropdown(
                            months, 
                            label="Month", 
                            info="Select the month of the transaction"
                        ),
                        gr.Slider(label="Hour", info="Enter the Hour in which the Transaction Occurred", minimum=0, maximum=23, step=1),
                        gr.Slider(label="Age", minimum=10, maximum=100, step=1),
                        gr.Radio(label="Gender", choices=gender_options),
                        gr.Dropdown(
                            options, 
                            label="Category", 
                            info="Select the Category of Purchase"
                        )
                        #gr.Radio(label="Time of Day",info="Enter the Time of Day in which the Transaction Occurred: \n hahaha", choices=time_of_day_options)
                    ],
                    outputs=[
                        gr.Text(label="Pobability Score", info="The probability that the transaction is fraud"),
                        gr.Text(label="Transaction is:")
                    ])

iface.launch()
