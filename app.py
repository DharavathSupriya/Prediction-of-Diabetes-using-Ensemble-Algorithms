import os
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def man():
    return render_template('home.html')
@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')
@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/', methods=['POST'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    data5 = request.form['e']
    data6 = request.form['f']
    data7 = request.form['g']
    data8 = request.form['h']
    # data3, data7,
    arr = np.array([[data1, data2, data4, data5, data6, data8]])
    pred = model.predict(arr)
    return render_template('after.html', data=pred)


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5001))
    app.run(port=port, debug=True, use_reloader=False)
