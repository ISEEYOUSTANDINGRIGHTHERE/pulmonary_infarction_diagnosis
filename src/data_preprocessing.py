import pandas as pd
#im loading the csv file from the url
def load_data(url):
    url="https://stanfordaimi.azurewebsites.net/datasets/3a7548a4-8f65-4ab7-85fa-3d68c9efc1bd/path-to-dataset.csv"
    data = pd.read_csv(url)
    return data

def preprocess_data(data):
    #we're preprocessing the data steps
    data = data.dropna() #dropna exlcudes all the missing values from the dataset we've acquired
    data = pd.get_dummies(data) #Each category level gets its own column, with a 1 indicating the presence of that category and a 0 indicating its absence.
    return data

if __name__== "__main__":
    raw_data_url = "https://stanfordaimi.azurewebsites.net/datasets/3a7548a4-8f65-4ab7-85fa-3d68c9efc1bd/path-to-dataset.csv"
    processed_data_path = "data/processed/processed_data.csv"

    data = load_data(raw_data_url)
    data = preprocess_data(data)
    data.to_csv(processed_data_path, index=False)
