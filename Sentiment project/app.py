import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from flask import Flask,render_template,url_for,request

lemmatizer = WordNetLemmatizer()

def lemma(doc):
    ans = [lemmatizer.lemmatize(text) for text in doc]
    return ans

model = pickle.load(open("logistic_model.pkl","rb"))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/prediction',methods=['POST'])
def prediction():
    if request.method=='POST':
        temp = request.form['message']
        final = model.predict([temp])
        # ans = int(str(temp))
        return render_template('prediction.html',ans=final)

if __name__ == "__main__":
    app.run(debug=True,use_reloader=False)