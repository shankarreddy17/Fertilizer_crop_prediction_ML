from flask import Flask,render_template,url_for,request
import joblib
import numpy as np

app=Flask(__name__)

model_path = 'Trained_Model/model.jb'
model = joblib.load(
    open(model_path, 'rb'))

@app.route('/')

def home():
    return render_template('home.html')

@app.route('/result',methods=['POST'])
def predict():
    
    Temparature = int(request.form['Temparature'])
    Humidity = int(request.form['Humidity'])
    Moisture = int(request.form['Moisture'])
    Soil_Type = int(request.form['Soil_Type'])
    Crop_Type = int(request.form['Crop_Type'])
    #Asthma = float(request.form['Asthma'])
    Nitrogen = float(request.form['Nitrogen'])
    Potassium = float(request.form['Potassium'])
    Phosphorous = float(request.form['Phosphorous'])
    

    query = np.array([[Temparature, Humidity, Moisture, Soil_Type, Crop_Type, 
                        Nitrogen, Potassium, Phosphorous]])

    prediction = model.predict(query)

    
    return render_template('result.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
