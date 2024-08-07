from flask import Flask, request, render_template
import joblib
import pandas as pd

app=Flask(__name__)
model=joblib.load('random_forest_model.pkl')
training_columns=joblib.load('training_columns.pkl')

def preprocess_new_data(new_data,training_columns):
    new_data=pd.get_dummies(new_data)
    missing_cols= set(training_columns) - set(new_data.columns)
    for col in missing_cols:
        new_data[col]=0
    new_data=new_data[training_columns]
    return new_data

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data=request.form.to_dict()
        data_df=pd.DataFrame([data])

        #now preprocess it..
        preprocessed_data=preprocess_new_data(data_df, training_columns)
        prediction = model.predict(preprocessed_data)[0] #thhis is to make a prediction
        prediction_text='Presenceof Pulmonary Infarction' if prediction == 1 else 'Absence of Pulmonary Infarction'
        
        return render_template('result.html', prediction=prediction_text)
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)