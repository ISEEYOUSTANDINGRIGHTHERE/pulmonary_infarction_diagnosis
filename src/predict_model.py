import joblib
import pandas as pd

model=joblib.load('random_forest_model.pkl')  # load the model from the file

def preprocess_new_data(new_data, training_columns):
    new_data=new_data.dropna()
    new_data=pd.get_dummies(new_data)

    missing_cols =set(training_columns) - set(new_data.columns)  #checks if 
    for col in missing_cols:
        new_data[col]=0
    new_data= new_data[training_columns] #reordering coloumns to match training data
    return new_data    

def make_predictions(new_data):
    training_columns = joblib.load('training_columns.pkl')  # Load training columns
    new_data = preprocess_new_data(new_data, training_columns)
    predictions = model.predict(new_data)
    return predictions

if __name__ == "__main__":
    # Example new patient data
    new_patient_data_path = r"D:\\PulmonaryInfarction\\multimodalpulmonaryembolismdataset\\processed_multimodalpulmonaryembolismdataset.csv"
    new_patient_data = pd.read_csv(new_patient_data_path)
    
    predictions = make_predictions(new_patient_data)
    print(predictions)
