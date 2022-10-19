from flask import Flask, render_template
import pickle
import numpy as np
import sklearn
from flask import request


model = pickle.load(open('rf-model.pkl','rb'))
app = Flask(__name__)


@app.route('/')
def hello_world():  # put application's code here
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_Investment():
    Credit_Mix = int(request.form.get('Credit_Mix'))
    Payment_Behviour = int(request.form.get('Payment_Behviour'))
    Payment_of_Min_Amount = int(request.form.get('Payment_of_Min_Amount'))
    Occupation = int(request.form.get('Occupation'))
    
    Credit_Score = model.predict(np.array([Credit_Mix,Payment_Behviour,Payment_of_Min_Amount,Occupation]).reshape(1,4))

    if Credit_Score[0] == 0:
        Credit_Score = 'You may have a poor credit score'
    elif Credit_Score[0] ==1:
        Credit_Score = 'You may have a standard credit score' 
    else:
        Credit_Score = 'You may have a good credit score!!!'

    return render_template('index.html', result=Credit_Score)


if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 8080)