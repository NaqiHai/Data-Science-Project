from fastapi import FastAPI
import numpy as np
import pickle
import uvicorn

app = FastAPI(debug=True)


@app.get('/')
def home():
    return {'text': 'Predicted No of Teacher is '}


@app.get('/predict')
def predict(INSTITUTION: int, ENROLLMENTS: int):
    model = pickle.load(open('model.pkl', 'rb'))
    prediction = model.predict([[INSTITUTION, ENROLLMENTS]])
    prediction = np.round(prediction)
    output = '{0}'.format(prediction)
    return {'THE TEACHER REQUIRED IS{}'.format(output)}


if __name__ == '__main__':
    uvicorn.run(app)
