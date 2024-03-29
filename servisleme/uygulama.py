from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__, template_folder="templates")

def load_models(modelName):
    if modelName == "Random Forest":  
        rf_model = joblib.load("evler_random_forest_model.pkl")
        return rf_model
    else:
        return None

def create_prediction_value(banyo_sayısı, kat_sayısı, Net_Metrekare, bina_yaşı, brüt_metrekare, ısıtma_tipi, oda_sayısı, ilce):
    res = pd.DataFrame(data={'Banyo Sayısı': [banyo_sayısı], 'Binanın Kat Sayısı': [kat_sayısı], 'Binanın Yaşı': [bina_yaşı],
                             'Brüt Metrekare': [brüt_metrekare], 'Isıtma Tipi': [ısıtma_tipi], 'Net Metrekare': [Net_Metrekare],
                             'Oda Sayısı': [oda_sayısı], 'ilce': [ilce]})
    return res

def predict_models(model, res):
    result = int(model.predict(res))
    return result

@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        selected_Banyo_Sayısı = int(request.form["banyo_sayisi"])
        selected_Binanın_Kat_Sayısı = int(request.form["kat_sayisi"])
        selected_Binanın_Yaşı = int(request.form["bina_yasi"])
        selected_Brüt_Metrekare = float(request.form["brut_metrekare"])
        selected_Isıtma_Tipi = int(request.form["isitma_tipi"])
        selected_Net_Metrekare = float(request.form["net_metrekare"])
        selected_Oda_Sayısı = int(request.form["oda_sayisi"])
        selected_ilce = int(request.form["ilce"])
        selected_model = request.form["model"]

        prediction_value = create_prediction_value(selected_Banyo_Sayısı, selected_Binanın_Kat_Sayısı, selected_Net_Metrekare,
                                                   selected_Binanın_Yaşı, selected_Brüt_Metrekare, selected_Isıtma_Tipi,
                                                   selected_Oda_Sayısı, selected_ilce)
        prediction_model = load_models(selected_model)

        if prediction_model:
            result = predict_models(prediction_model, prediction_value)
            return render_template("result.html", result=result)
        else:
            return render_template("error.html", message="Model yüklenirken hata oluştu.")

if __name__ == "__main__":
    app.run(debug=True)
