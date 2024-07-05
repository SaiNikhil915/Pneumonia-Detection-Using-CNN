from flask import Flask, render_template, request
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

app = Flask(__name__)


@app.route("/")
def home():
    return render_template('home.html')

@app.route("/pneumonia", methods=['GET', 'POST'])
def pneumoniaPage():
    return render_template('pneumonia.html')

@app.route("/pneumoniapredict", methods=['POST'])
def pneumoniapredictPage():
    if request.method == 'POST':
        try:
            if 'image' in request.files:
                img = Image.open(request.files['image']).convert('L')
                img = img.resize((36, 36))
                img = np.asarray(img)
                img = img.reshape((1, 36, 36, 1))
                img = img / 255.0
                model = load_model("models/pneumonia.h5")
               
                pred = np.argmax(model.predict(img)[0])
                return render_template('pneumonia_predict.html', pred=pred)
            else:
                message = "Please upload an image."
                return render_template('pneumonia.html', message=message)
            
        except:
            message = "An error occurred. Please try again."
            return render_template('pneumonia.html', message=message)

if __name__ == '__main__':
    app.run(debug=True)
