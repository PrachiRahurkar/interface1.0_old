from flask import Flask,request,jsonify,render_template
import torch
import pandas as pd
from bert_predict import get_result
# Use torch to load in the pre-trained model

# Initialise the Flask app
app = Flask(__name__, static_url_path='/static')

# Set up the main route
@app.route('/predict',methods=['POST'])
def predict():
    # Extract the input
    para = request.form['para']
    qstn = request.form['qstn']
    print(para)
    print(qstn)
    ans = get_result(qstn, para)
    print(ans)
    f = open("perturbs.txt", "a")
    temp = (qstn, para, ans)
    f.write(temp)
    f.close()
    return ans

@app.route('/')
def render_page1():
    return render_template('page1.html')

@app.route('/tutorial')
def render_tutorial():
    return render_template('tutorial.html')

@app.route('/samples')
def render_samples():
    return render_template('samples.html')

@app.route('/adversary_ex')
def render_adversary_ex():
    return render_template('adversary_ex.html')

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=8000)