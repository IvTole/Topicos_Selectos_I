from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

def BinaryClassifierDFPrep(df,input_cols,target_var,treat_outliers:bool=False,treat_neg_values:bool=False,scaling:bool=False, use_pca:bool=False, dim:int=1):
    """
    :return: a pair (X, y) where X is the data frame containing all attributes and y is the corresping series of class values
    """

    # variable 'education' pre-processing
    if 'education' in input_cols:
        df['education'] = np.where(df['education'] == "basic.4y", "Basic", df["education"])
        df['education'] = np.where(df['education'] == "basic.6y", "Basic", df["education"])
        df['education'] = np.where(df['education'] == "basic.9y", "Basic", df["education"])
        df['education'] = np.where(df['education'] == "high.school", "High School", df["education"])
        df['education'] = np.where(df['education'] == "professional.course", "Professional Course", df["education"])
        df['education'] = np.where(df['education'] == "university.degree", "University Degree", df["education"])
        df['education'] = np.where(df['education'] == "illiterate", "Illiterate", df["education"])
        df['education'] = np.where(df['education'] == "unknown", "Unknown", df["education"])

    # numeric and categorical data lists
    categorical_data = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_data = df.select_dtypes(include=['number']).columns.tolist()

    # categorical data to dummy variables
    for category in categorical_data:
        dummy_df = pd.get_dummies(df[category], prefix=category)
        df = df.join(dummy_df)
    df.drop(categorical_data, inplace=True, axis=1)

    # scaling
    if scaling:
        # Scaling data
        scaler = StandardScaler()
        X_std = scaler.fit_transform(df[numeric_data])
        print("Data has been scaled: StandardScaler")
        if use_pca:
            pca = PCA(n_components=dim)
            X_pca = pca.fit_transform(X_std)
            df.drop([element for element in numeric_data if element!=target_var], inplace=True, axis=1)
            df_pca_dummy = pd.DataFrame()
            for i in range(0,X_pca.shape[1]):
                df_pca_dummy['Component'+str(i)] = X_pca[:,i]
            df = df.join(df_pca_dummy)
            print(df.columns)
        

    # New input cols
    input_cols = [v for v in df.columns if v!=target_var]

    # data frames containing attributes(X) and target(y)
    X = df[input_cols]
    y = df[target_var]

    print(X.head())


    return X, y