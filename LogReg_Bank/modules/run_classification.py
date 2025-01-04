from module_data_path import csv_data_path
from module_data_load import load_data_frame

df_path = csv_data_path()
df = load_data_frame(path=df_path)

print("Shape = ", df.shape)