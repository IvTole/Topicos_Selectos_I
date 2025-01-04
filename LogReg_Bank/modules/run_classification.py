#!/usr/bin/env python3  # Optional: for Unix/Linux environments

from datetime import datetime

# own external modules
from module_data_path import csv_data_path
from module_data_load import load_data_frame

# Main function
def main():
    # get current date and time
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("Start date & time : ", start_datetime)

    df_path = csv_data_path()
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

    print("Shape = ", df.shape)

# Main Execution Block: Code that runs when the script is executed directly
if __name__ == '__main__':
    main()