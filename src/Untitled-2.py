def load_data(path):
    data = pd.read_csv("D:\\PulmonaryInfarction\\multimodalpulmonaryembolismdatase\\ICD.csv")
    print(data.dtypes)  # Print the data types of each column
    return data
    