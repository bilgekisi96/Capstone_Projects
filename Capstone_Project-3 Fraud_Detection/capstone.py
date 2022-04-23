import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
from flask import Flask, render_template, request
from sklearn.preprocessing import RobustScaler
from playsound import playsound
import joblib

app = Flask(__name__)

def prediction(amount, v3, v4, v7, v10, v11, v12, v14, v16, v17, model='rf_model_day.pkl'):

    new_df = {'V3': float(v3),
              'V4':float(v4),
              'V7':float(v7),
              'V10':float(v10),
              'V11':float(v11),
              'V12':float(v12),
              'V14':float(v14),
              'V16':float(v16),
              'V17':float(v17),
              'Amount':float(amount)}
    features = pd.DataFrame(new_df, index=[0])
    
    # final_scaler = pickle.load(open(scaler, 'rb'))
    # scaler = RobustScaler()
    # data = pd.read_csv(df)
    # scaler.fit(data)

    rf_model = joblib.load(open(model, 'rb'))

    # use_df = final_scaler.transform(features)

    # final_model = load_model(model)

    prediction = rf_model.predict(features)
    # d = prediction[0][0]
    # return round(d,2)
    return prediction

@app.route('/', methods=['POST', 'GET'])
def rootpage():
    res = None
    ses = None
    amount = ''
    v3 = ''
    v4 = ''
    v7 = ''
    v10 = ''
    v11 = ''
    v12 = ''
    v14 = ''
    v16 = ''
    v17 = ''
    if request.method == 'POST':
        amount = request.form.get('amount')
        v3 = request.form.get('v3')
        v4 = request.form.get('v4')
        v7 = request.form.get('v7')
        v10 = request.form.get('v10')
        v11 = request.form.get('v11')
        v12 = request.form.get('v12')
        v14 = request.form.get('v14')
        v16 = request.form.get('v16')
        v17 = request.form.get('v17')
        res = prediction(amount, v3, v4, v7, v10, v11, v12, v14, v16, v17)
        ses = int(res)
        if ses == 0:
            playsound('success.wav', False)
        else:
            playsound('alarm.wav', False)

    return render_template('index.html', res=res, ses=ses, amount=amount, v3=v3, v4=v4, v7=v7, v10=v10, v11=v11, v12=v12, v14=v14, v16=v16, v17=v17)

app.run()
# app.run(host='0.0.0.0', port=5000, debug=True)
