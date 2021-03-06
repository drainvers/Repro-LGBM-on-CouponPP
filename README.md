# Notes

This is an implementation of [another recommender 
system](https://github.com/ouwyukha/Repro-RF-XGB-on-CouponPP) created 
using LightGBM for benchmarking purposes.

The Kaggle competition used to evaluate this model is Recruit Ponpare's 
[Coupon Purchase 
Prediction](https://www.kaggle.com/c/coupon-purchase-prediction/)

For the preprocessed data, please refer to the original repository.

# Latest Results (MAP@10)
- Private score: 0.00562
- Public score: 0.00611

# How to run
This assumes you are in a directory with the following structure (and are using the Coupon Purchase Prediction dataset):
```
current_directory
|-> dataset
|   |-> *all dataset csv files go here*
|
|-> translation
|   |-> *translation mapping csv files go here*
|
|-> generate_positive_negative_samples.py
|-> generate_positive_negative_samples_no_translation.py
|-> CPP_REPRO_LGBM.ipynb
```
1. First, run `generate_positive_negative_examples.py` (a variant of this script that uses the dataset without translation is also provided)
2. Once positive and negative examples have been generated, you will get the following files:
    ```
    current_directory/CPP_REPRO_coupon_detail_train.csv
    current_directory/CPP_REPRO_coupon_user_list.csv
    current_directory/CPP_REPRO_coupon_list_train.csv
    current_directory/CPP_REPRO_coupon_list_train.csv
    current_directory/CPP_REPRO_coupon_list_test.csv
    current_directory/positive_coupons_train.csv
    current_directory/negative_coupons_train.csv
    ```
3. Run the IPython notebook
3. The generated submission CSV file can then be uploaded to Kaggle