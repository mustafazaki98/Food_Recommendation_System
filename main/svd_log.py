import os, sys
sys.path.append(os.path.abspath('/Users/mustafazaki/Downloads/Food Recommendation System/preprocessing'))
sys.path.append(os.path.abspath('/Users/mustafazaki/Downloads/Food Recommendation System/models'))

import numpy as np
import pandas as pd
from preprocessing import LogNormaliser
from svd import SVD



def main():
    log_svd = LogNormaliser()
    preprocessed_df, names = log_svd.fit_transform()

    # processed_df = data_preprocessing(df, names)
    svd_model = SVD(preprocessed_df)
    X_train, X_val = svd_model.train_test_split()
    svd_model.fit(X_train, X_val)
    # svd_model.save(filename='/Users/mustafazaki/Downloads/Food Recommendation System/models/saved models/svd_zscore.pkl')


if __name__ == '__main__':
    main()
