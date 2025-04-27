from flask import Flask, request, render_template, redirect, url_for
import joblib
import pandas as pd
import os

app = Flask(__name__)
model = joblib.load("C:\\Users\\Akash\\OneDrive\\Documents\\GitHub\\pulmonary_infarction_diagnosis\\src\\random_forest_model.pkl")
training_columns = joblib.load('training_columns.pkl')

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def preprocess_new_data(new_data, training_columns):
    new_data = pd.get_dummies(new_data)
    missing_cols = set(training_columns) - set(new_data.columns)
    for col in missing_cols: 
        new_data[col] = 0
    new_data = new_data[training_columns]
    return new_data

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Handle file upload
        if 'image' not in request.files:
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            data = request.form.to_dict()
            data['image_url'] = url_for('static', filename='uploads/' + file.filename)
            data_df = pd.DataFrame([data])
            preprocessed_data = preprocess_new_data(data_df, training_columns)
            prediction = model.predict(preprocessed_data)[0]
            prediction_text = 'Presence of Pulmonary Infarction' if prediction == 1 else 'Absence of Pulmonary Infarction'
            return render_template('result.html', prediction_text=prediction_text, image_url=data['image_url'],
                                   patient_id=data['patient_id'], age=data['age'], sex=data['sex'])
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
