import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer


from module_path import csv_data_path

COL_INVOICE_NO = 'InvoiceNo'
COL_STOCK_CODE = 'StockCode'
COL_DESCRIPTION = 'Description'
COL_QUANTITY = 'Quantity'
COL_INVOICE_DATE = 'InvoiceDate'
COL_UNIT_PRICE = 'UnitPrice'
COL_CUSTOMER_ID = 'CustomerID'
COL_COUNTRY = 'Country'

COL_RECENCY = 'Recency'
COL_FREQUENCY = 'Frequency'
COL_AMOUNT = 'Amount'
COL_DIFF = 'Diff'

class Dataset:

    def __init__(self, num_samples: int = None, random_seed: int = 42):
        """
        :param num_samples: the number of samples to draw from the data frame; if None, use all samples
        :param random_seed: the random seed to use when sampling data points
        """

        self.num_samples = num_samples
        self.random_seed = random_seed
    
    def load_data_frame(self) -> pd.DataFrame:
        """
        :return: the full data frame for this dataset (including the class column)

        Note: Null values are dropped and invoice date variable type is changed to timestamp
        """
        df = pd.read_csv(csv_data_path(), sep = ",", encoding = "ISO-8859-1").dropna()
        df[COL_INVOICE_DATE] = pd.to_datetime(df[COL_INVOICE_DATE], format = '%m/%d/%Y %H:%M')

        if self.num_samples is not None:
            df = df.sample(self.num_samples, random_state=self.random_seed)        
        return df
    
    def load_data_frame_rfm(self) -> pd.DataFrame:
        """
        :return: the data frame containing the RFM variables

        Note: outliers are removed from the dataframe (iqr based on 5 and 95 percentile)
        """
        df = self.load_data_frame()

        # Amount
        df[COL_AMOUNT] = df[COL_QUANTITY]*df[COL_UNIT_PRICE]
        df_m = df.groupby(COL_CUSTOMER_ID)[COL_AMOUNT].sum().reset_index()

        # Frequency
        df_f = df.groupby(COL_CUSTOMER_ID)[COL_INVOICE_NO].count().reset_index()

        # Recency
        max_date = max(df[COL_INVOICE_DATE])
        df[COL_DIFF] = max_date - df[COL_INVOICE_DATE]
        df_r = df.groupby(COL_CUSTOMER_ID)[COL_DIFF].min().reset_index()
        df_r[COL_DIFF] = df_r[COL_DIFF].dt.days

        # RFM
        df_rfm = pd.merge(pd.merge(df_r,df_f,on=COL_CUSTOMER_ID,how='inner'), df_m, on=COL_CUSTOMER_ID, how='inner')
        df_rfm.columns = [COL_CUSTOMER_ID, COL_RECENCY, COL_FREQUENCY, COL_AMOUNT]

        # Remove outliers
        col_outliers = [COL_RECENCY, COL_FREQUENCY, COL_AMOUNT]
        for col in col_outliers:
            lower_bound = df_rfm[col].quantile(0.05)
            upper_bound = df_rfm[col].quantile(0.95)
            iqr = upper_bound - lower_bound
            df_rfm = df_rfm[(df_rfm[col] >= lower_bound - 1.5*iqr) & \
                           (df_rfm[col] <= upper_bound + 1.5*iqr)]

        return df_rfm
    
    def load_data_frame_rfm_stdscale(self):
        """
        :return: the data frame containing the RFM variables (with standard scale)
        """
        df_rfm = self.load_data_frame_rfm()
        features =[COL_RECENCY, COL_FREQUENCY, COL_AMOUNT]

        scaler = StandardScaler()
        df_rfm[features] = scaler.fit_transform(df_rfm[features])

        return df_rfm
    
    def load_data_frame_rfm_minmaxscale(self):
        """
        :return: the data frame containing the RFM variables (with minmax scale)
        """
        df_rfm = self.load_data_frame_rfm()
        features =[COL_RECENCY, COL_FREQUENCY, COL_AMOUNT]

        scaler = MinMaxScaler(feature_range=(0,1))
        df_rfm[features] = scaler.fit_transform(df_rfm[features])

        return df_rfm