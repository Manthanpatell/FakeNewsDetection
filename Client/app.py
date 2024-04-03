from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle



#instance of Flask
app = Flask(__name__) 

#loading the pre-trained model and vectorizer through pickle
loaded_model = pickle.load(open('models/model.pkl', 'rb'))
load_tfvect = pickle.load(open('models/tfvect.pkl', 'rb'))


def fake_news_det(news):
    input_data = [news]
    vectorized_input_data = load_tfvect.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
    return prediction

#route for frontend 
@app.route('/')
def home():
    return render_template('index.html')

#Gets data from form and processes and returns the response back to frontend
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['content']
        pred = fake_news_det(message)
        print(pred)
        if pred == 'fake':
            result = "Fake news detected!!"
        elif pred == 'true':
            result = "Hurayyyy! News is real"
        return render_template('index.html', prediction=pred, message=message, result=result)
    
if __name__ == '__main__':
    app.run(debug=True)
 