# -*- coding: utf-8 -*-
"""
API to suggest tags when creating a new question to Stack Overflow

Created on Thursday Aug 27 2020

@author: Sylvain Friot
"""

import joblib
import numpy as np
import pandas as pd
import skmultilearn
import tensorflow_hub as hub

from flask import Flask, render_template, request
from bs4 import BeautifulSoup

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)
app = Flask(__name__)
my_pipeline = joblib.load("ml05_pipeline.joblib")
list_labels = joblib.load("ml05_labels.joblib")

def get_labels(body, title):
    inter_body = BeautifulSoup(body, features="html.parser")
    for code_to_remove in inter_body.find_all("code"):
        code_to_remove.decompose()
    clean_body = inter_body.get_text().replace("\n", " ")
    inter_title = BeautifulSoup(title, features="html.parser")
    clean_title = inter_title.get_text().replace("\n", " ")
    X_both = clean_title + " " + clean_body
    my_data = np.array(model([X_both]))
    my_predict = my_pipeline.predict(my_data)
    labels = []
    for cpt in range(my_predict.shape[1]):
        if my_predict[0, cpt] == 1:
            labels.append(list_labels[cpt])
    return {"body": clean_body, "title": clean_title, "labels": labels}

@app.route("/", methods=["GET", "POST"])
def labels_questions():
    if request.method == "POST":
        title = request.form["title"]
        body = request.form["body"]
        result = get_labels(body, title)
        return render_template("labels.html", results=result)
    else:
        return render_template("welcome.html")

if __name__ == "__main__":
    app.run()