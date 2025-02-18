#!/usr/bin/env python3  # Optional: for Unix/Linux environments

from datetime import datetime

# scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# own external modules
from module_data_path import csv_data_path, model_data_path
from module_data_load import load_data_frame
from module_data_preprocessing import BinaryClassifierDFPrep
from module_model_evaluation import model_evaluate


# Main function
def main():
    # get current date and time
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("Start date & time : ", start_datetime)

    df_path = csv_data_path()
    #model_path = model_data_path()
    df = load_data_frame(path=df_path)

    # input and target variables

    # available variables
    col_list = ['age', 'job', 'marital', 'education', 'default', 'housing',
                'loan', 'contact', 'month', 'day_of_week', 'duration', 'campaign',
                'pdays', 'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx',
                'cons.conf.idx', 'euribor3m', 'nr.employed']
    input_cols = col_list
    target_var = 'y'
    print("Input variables: ", input_cols)
    print("Target variable: ", target_var)

    # Data pre processing
    X,y = BinaryClassifierDFPrep(df=df,
                                 input_cols=input_cols,
                                 target_var=target_var,
                                 treat_neg_values=False,
                                 treat_outliers=False,
                                 scaling=True)
    
    # Data split in training and test dataframes
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3, shuffle=True)
    print("Data splitting")

    # Evaluate models
    model_evaluate(LogisticRegression(solver='lbfgs', max_iter=3000), X_train, y_train, X_test, y_test)

# Main Execution Block: Code that runs when the script is executed directly
if __name__ == '__main__':
    main()