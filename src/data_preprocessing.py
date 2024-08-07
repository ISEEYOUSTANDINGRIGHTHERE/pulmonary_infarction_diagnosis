import pandas as pd
#im loading the csv file from the SDD
def load_data(path):   # path is a variable do not include path link
    data = pd.read_csv(path)
    return data

def preprocess_data(data):
    #we're preprocessing the data steps
    data = data.dropna() #dropna exlcudes all the missing values from the dataset we've basically downloaded
    data = pd.get_dummies(data) #Each category level gets its own column, with a 1 indicating the presence of that category and a 0 indicating its absence.
    return data

if __name__== "__main__":
    raw_data_path = r"D:\\PulmonaryInfarction\multimodalpulmonaryembolismdataset\ICD.csv" #can use path to csv outformat->check
    processed_data_path = r"D:\\PulmonaryInfarction\\multimodalpulmonaryembolismdataset\\processed_multimodalpulmonaryembolismdataset.csv"

    data = load_data(raw_data_path)
    data = preprocess_data(data)
    data.to_csv(processed_data_path, index=False)
