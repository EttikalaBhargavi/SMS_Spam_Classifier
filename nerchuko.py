

#importing the essential libraries
from flask import Flask , render_template , request
import pickle
import numpy as np

#loading the saved models
model = pickle.load(open('model.pkl' , 'rb'))
cv = pickle.load(open('vector.pkl' , 'rb'))

#creating flask instance
app = Flask(__name__)

#home page flask code
@app.route('/')
def home():
  return render_template('home.html')


#prediction page flask code
@app.route('/predict' , methods=['POST'])
def predict():
  message = request.form['text']
  data = [message]
  data = cv.transform(data).toarray()
  pred = model.predict(data)
  return render_template('result.html' , prediction = pred)


#Running the flask app
if __name__ == '__main__':
  app.run(debug=True)
