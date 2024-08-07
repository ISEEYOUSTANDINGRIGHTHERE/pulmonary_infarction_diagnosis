import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

def load_data(path):
    data=pd.read_csv(r"D:\\PulmonaryInfarction\\multimodalpulmonaryembolismdataset\\processed_multimodalpulmonaryembolismdataset.csv")
    return data

def split_data(data):
    X = data.drop('DISEASES OF PULMONARY CIRCULATION:presence', axis=1)  # Replace 'target_column' with your target column name
    y = data['DISEASES OF PULMONARY CIRCULATION:presence']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    return accuracy, cm, roc_auc

if __name__=="__main__":
    processed_data_path=r"D:\\PulmonaryInfarction\\multimodalpulmonaryembolismdataset\\processed_multimodalpulmonaryembolismdataset.csv"
    data=load_data(processed_data_path)
    X_train,X_test,y_train,y_test = split_data(data)

    model=train_model(X_train, y_train)
    accuracy,cm,roc_auc= evaluate_model(model,X_test,y_test)

    print(f"Accuracy: {accuracy}")
    print(f"Confusion Matrix: \n{cm}")
    print(f"ROC AUC Score: {roc_auc}")