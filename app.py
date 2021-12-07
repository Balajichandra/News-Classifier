from flask import Flask
from flask import render_template,request
import pickle
import numpy as np

#self._vectorizer = CountVectorizer()
# Load the Random Forest CLassifier model

filename = 'newsClassification.pkl'
classifier = pickle.load(open(filename, 'rb'))
cv = pickle.load(open('cv-transform1.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_pred = classifier.predict(vect)
        return render_template('result.html',prediction=my_pred)
if __name__ == "__main__":
    app.run(debug=True)        