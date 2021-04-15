import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Flask
from flask import Flask, redirect, url_for, request, render_template, jsonify
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
from utils import *
from sklearn.linear_model import LinearRegression
import seaborn as sns


import time

# Define a flask app
app = Flask(__name__)
PEOPLE_FOLDER = os.path.join('static', 'image_folder')
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER

# Model saved

@app.route('/')
def home():
    # Main page
    return render_template('index.html')


@app.route('/upload', methods = ['POST', 'GET'])
def upload_file():
    if request.method == 'POST':
        if request.files:
            f = request.files['file']
            f.save(os.path.join(app.config['UPLOAD_FOLDER'],f.filename))
            print("Saved")
        return redirect(request.url)

    return render_template('index.html')


@app.route('/predict', methods=['POST'])


def predict():
    df_new = processGndData(os.path.join(app.config['UPLOAD_FOLDER'],'GNR631_GroundTruth.xlsx'))
    dfd_new, n_days, n_plots = processDroneData(df_new, os.path.join(app.config['UPLOAD_FOLDER'],'LAI_upto_tasseling.xlsx'))
    print(dfd_new)

    t = time.time()

    path1 = os.path.join(app.config['UPLOAD_FOLDER'],f'{t}_data.png')
    save_LBPLot(df_new, path1)
    path2 = os.path.join(app.config['UPLOAD_FOLDER'],f'{t}_reg.png')
    save_RegPlot(df_new, path2)
    path3 = os.path.join(app.config['UPLOAD_FOLDER'],f'{t}_comp_reg.png')
    save_CompReg(df_new,path3)
    path4 = os.path.join(app.config['UPLOAD_FOLDER'],f'{t}_sla.png')
    save_SLAGauss(df_new,path4)
    path5 = os.path.join(app.config['UPLOAD_FOLDER'],f'{t}_dsscatter.png')
    save_DScatter(dfd_new, path5, n_days, n_plots)
    path6 = os.path.join(app.config['UPLOAD_FOLDER'],f'{t}_dslbscatter.png')
    save_DLBScatter(dfd_new, path6, n_days, n_plots)
    path7 = os.path.join(app.config['UPLOAD_FOLDER'],f'{t}_dssbscatter.png')
    save_DSBScatter(dfd_new, path7,n_days, n_plots)
    path8 = os.path.join(app.config['UPLOAD_FOLDER'],f'{t}_dsslascatter.png')
    save_DSLAScatter(dfd_new, path8,n_days, n_plots)
    path9 = os.path.join(app.config['UPLOAD_FOLDER'],f'{t}_dagbscatter.png')
    save_DAGBScatter(dfd_new, path9,n_days, n_plots)
    path10 = os.path.join(app.config['UPLOAD_FOLDER'],f'{t}_dagbrscatter.png')
    save_DAGBRScatter(dfd_new, path10,n_days, n_plots)



    return render_template('result.html',data_plot=path1, reg_plot=path2, comp_reg_plot=path3, sla_plot=path5, plb_plot=path6, psb_plot=path7, dsla_plot=path8,
                            agb_plot=path9, agbr_plot=path10)



if __name__ == '__main__':
    app.run(debug=True)
