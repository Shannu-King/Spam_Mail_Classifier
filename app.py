from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import os


app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
mail_datas = pd.read_csv(os.path.join(BASE_DIR, 'mail_data.csv'))
mails = mail_datas.where((pd.notnull(mail_datas)), '')
mails.loc[mails['Category'] == 'spam', 'Category'] = 0
mails.loc[mails['Category'] == 'ham', 'Category'] = 1
x = mails['Message']
y = mails['Category']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
x_train_features = feature_extraction.fit_transform(x_train)
y_train = y_train.astype('int')
model = LogisticRegression()
model.fit(x_train_features, y_train)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        message = request.form["message"]
        data = feature_extraction.transform([message])
        prediction = model.predict(data)
        result = "Spam" if prediction[0] == 0 else "Ham"
        return render_template("index.html", prediction=result)
    return render_template("index.html", prediction="")

if __name__ == "__main__":
    app.run(host="0.0.0.0")

