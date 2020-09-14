#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest CLassifier model
filename = 'Life_expectancy.pkl'
xgb = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        AdultMortality = float(request.form['AdultMortality'])
        Alcohol = float(request.form['Alcohol'])
        HepatitisB = float(request.form['HepatitisB'])
        Measles = int(request.form['Measles'])
        BMI = float(request.form['BMI'])
        under_fivedeaths = int(request.form['under-fivedeaths'])
        Polio = float(request.form['Polio'])
        Totalexpenditure = float(request.form['Totalexpenditure'])
        Diphtheria = float(request.form['Diphtheria'])
        HIV_AIDS = float(request.form['HIV/AIDS'])
        GDP = float(request.form['GDP'])
        Population = float(request.form['Population'])
        Incomecompositionofresources = float(request.form['Incomecompositionofresources'])
        Schooling = float(request.form['Schooling'])
        
        data = np.array([[AdultMortality, Alcohol, HepatitisB, Measles, BMI,under-fivedeaths,Polio,Totalexpenditure,Diphtheria,
                          HIV/AIDS,GDP,Population,Incomecompositionofresources,Schooling]])
        my_prediction = xgb.predict(data)
        
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)

