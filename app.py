from flask import Flask, request, render_request
import numpy as np
import pickle

model = pickle.load(open('Logestic.pkl','r'))

app = Flask(__name__)

@app.route('/')
def man():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def home():
    arr = [int(x) for x in arr.split(',')]
    pred = model.predict(arr)

    return render_template('after.html', data=pred)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=False)  

