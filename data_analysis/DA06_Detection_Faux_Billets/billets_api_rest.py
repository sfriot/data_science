# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 23:39:19 2019

@author: Sylvain
"""

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

file_name = 'direct_logreg.joblib'
direct_logreg = joblib.load(file_name)
file_name = 'acp_acp.joblib'
acp_acp = joblib.load(file_name)
file_name = 'acp_logreg.joblib'
acp_logreg = joblib.load(file_name)

def notes_prediction(interdata, method):
    billet_id = interdata.index
    
    if (method == 'acp') | (method == 'cross'):  # calculates data projection with the pca, then scale the projected data and make a prediction
        acp_data = interdata[acp_acp.variables]
        acp_data_scaled = acp_acp.scaler.transform(acp_data)
        data_projected = pd.DataFrame(acp_acp.pca.transform(acp_data_scaled), index=billet_id, \
                columns = ["F{} ({}%)".format(i+1, round(100*acp_acp.pca.explained_variance_ratio_[i],1)) for i in np.arange(acp_acp.n_components)])
        data_projected = data_projected[acp_logreg.variables_explicatives]
        if acp_logreg.scaler is None:
            reg_data_scaled = np.array(data_projected)
        else:
            interx = np.array(data_projected)
            reg_data_scaled = acp_logreg.scaler.transform(interx)
        acp_prediction_proba = acp_logreg.classification.predict_proba(reg_data_scaled)[:,1]
        acp_prediction = acp_logreg.classification.predict(reg_data_scaled).astype(int)

    if (method == 'direct') | (method == 'cross'):  # calculates data projection with the pca, then scale the projected data and make a prediction
        reg_data = interdata[direct_logreg.variables_explicatives]
        if direct_logreg.scaler is None:
            data_scaled = np.array(reg_data)
        else:
            interx = np.array(reg_data)
            data_scaled = direct_logreg.scaler.transform(interx)
        direct_prediction_proba = direct_logreg.classification.predict_proba(data_scaled)[:,1]
        direct_prediction = direct_logreg.classification.predict(data_scaled).astype(int)
        
    output = []
    if method == 'cross':
        final_prediction = []
        for i in np.arange(len(billet_id)):
            if direct_prediction[i] == acp_prediction[i]:
                final_prediction.append(direct_prediction[i])
            else:
                final_prediction.append(-1)
        output = [{'id':billet_id[i],'resultat':int(final_prediction[i]),'direct':direct_prediction_proba[i],'acp':acp_prediction_proba[i]} for i in np.arange(len(billet_id))]
    elif method == 'direct':
        output = [{'id':billet_id[i],'resultat':int(direct_prediction[i]),'proba_true':direct_prediction_proba[i]} for i in np.arange(len(billet_id))]
    elif method == 'acp':
        output = [{'id':billet_id[i],'resultat':int(acp_prediction[i]),'proba_true':acp_prediction_proba[i]} for i in np.arange(len(billet_id))]
    return output
    
@app.route('/billets', methods=['GET', 'POST'])
def billets_upload():
    if request.method == 'POST':
        analyse_type = request.form['analyseType']
        csvfile = request.files['file']
        data = pd.read_csv(csvfile, index_col='id')
        result = notes_prediction(data, analyse_type)
        if analyse_type == 'cross':
            return render_template("results_cross.html", results = result)
        else:
            return render_template("results_simple.html", results = result)
    else:
        return render_template('welcome.html')

@app.route('/billets/<any(direct,acp,cross):analyse_type>', methods=['POST'])
def billets_json(analyse_type):
    if request.is_json:
        data = pd.DataFrame.from_dict(request.json)
        result = notes_prediction(data, analyse_type)
    else:
        result = "La requête doit être de type json"
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
