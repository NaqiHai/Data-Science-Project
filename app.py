from flask import Flask, request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def hello_world():
    return render_template("Sindh.html")


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final = [np.array(int_features)]
#    print(int_features)
 #   print(final)
    prediction = model.predict(final)
    prediction=np.round(prediction)
    output = '{0}'.format(prediction)
    return render_template('Sindh.html', pred='No of Teacher.\n Required is {}'.format(output),
                           bhai="kuch karna hain iska ab?")


if __name__ == '__main__':
    app.run(debug=True)
