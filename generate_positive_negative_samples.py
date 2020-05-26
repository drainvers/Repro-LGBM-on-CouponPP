import pandas as pd
import os
import numpy as np
import itertools
from tqdm import tqdm

tqdm.pandas()

def find_positive(row):
    row = row.iloc[0]
    query = dataframes['coupon_list_train'][(dataframes['coupon_list_train'].DISPFROM <= row.I_DATE) & (dataframes['coupon_list_train'].DISPEND >= row.I_DATE)]
    query = query[query.COUPON_ID_hash.isin(cvdict[row.USER_ID_hash])]
    query['USER_ID_hash'] = row.USER_ID_hash
    return query
    
def find_negative(row):
    row = row.iloc[0]
    user_data = dataframes['user_list'].loc[dataframes['user_list']['USER_ID_hash'] == row.USER_ID_hash].iloc[0]
    query = dataframes['coupon_list_train'][(dataframes['coupon_list_train'].DISPFROM <= row.I_DATE) & (dataframes['coupon_list_train'].DISPEND >= row.I_DATE)]
    query = query[(user_data.REG_DATE <= query.DISPEND) & (query.DISPFROM <= user_data.WITHDRAW_DATE)]
    query = query[~query.COUPON_ID_hash.isin(cvdict[row.USER_ID_hash])]
    query['USER_ID_hash'] = row.USER_ID_hash
    query = query.sample(n=9, random_state=0)
    return query

def list_all_files_in(dirpath):
    return [(os.path.splitext(filename)[0], os.path.join(dirname, filename))
            for dirname, _, filenames in os.walk(dirpath)
            for filename in filenames]

df_paths = dict(list_all_files_in('dataset'))
trans_paths = dict(list_all_files_in('translation'))
dataframes = {}
translations = {}

# Dataset
print('Loading dataset...')
for df_name, csv_path in df_paths.items():
    dataframes[df_name] = pd.read_csv(csv_path)
    print("Read from", csv_path, "into", df_name, "successful")

# Translation mappings
print('Loading translations...')
for df_name, csv_path in trans_paths.items():
    translations[df_name] = pd.read_csv(csv_path, delimiter=';', index_col='jpn')
    print("Read from", csv_path, "into", df_name, "successful")

# Rename some columns, this should make translation easier
dataframes['coupon_visit_train'].rename(columns={'VIEW_COUPON_ID_hash':'COUPON_ID_hash'}, inplace=True)
dataframes['coupon_list_train'].rename(columns={'large_area_name':'LARGE_AREA_NAME', 'ken_name':'PREF_NAME', 'small_area_name':'SMALL_AREA_NAME'},inplace=True)
dataframes['coupon_list_test'].rename(columns={'large_area_name':'LARGE_AREA_NAME', 'ken_name':'PREF_NAME', 'small_area_name':'SMALL_AREA_NAME'},inplace=True)

# Actual translation work
print('Translating columns and cleaning up...')
trans_column_dict = {
    'coupon_list_train': ['CAPSULE_TEXT', 'GENRE_NAME', 'LARGE_AREA_NAME', 'PREF_NAME', 'SMALL_AREA_NAME'],
    'coupon_list_test': ['CAPSULE_TEXT', 'GENRE_NAME', 'LARGE_AREA_NAME', 'PREF_NAME', 'SMALL_AREA_NAME'],
    'coupon_area_train': ['SMALL_AREA_NAME', 'PREF_NAME'],
    'coupon_area_test': ['SMALL_AREA_NAME', 'PREF_NAME'],
    'prefecture_locations': ['PREF_NAME', 'PREFECTUAL_OFFICE'],
    'user_list': ['PREF_NAME'],
    'coupon_detail_train': ['SMALL_AREA_NAME']
}

for df_name, columns in trans_column_dict.items():
    for col in columns:
        dataframes[df_name][col] = dataframes[df_name][col].replace(translations[col].to_dict()['en'])

for df_name in ['coupon_list_train', 'coupon_list_test']:
    dataframes[df_name]['VALIDFROM'].fillna(dataframes[df_name]['DISPFROM'], inplace=True)
    dataframes[df_name]['VALIDEND'].fillna(pd.Timestamp.max, inplace=True)

    dataframes[df_name]['DISPFROM'] = pd.to_datetime(dataframes[df_name]['DISPFROM'])
    dataframes[df_name]['DISPEND'] = pd.to_datetime(dataframes[df_name]['DISPEND'])
    dataframes[df_name]['VALIDFROM'] = pd.to_datetime(dataframes[df_name]['VALIDFROM'])
    dataframes[df_name]['VALIDEND'] = pd.to_datetime(dataframes[df_name]['VALIDEND'])

    dataframes[df_name]['VALIDPERIOD'].fillna((dataframes[df_name]['VALIDEND'] - dataframes[df_name]['VALIDFROM']) / np.timedelta64(1, 'D'), inplace=True)
    dataframes[df_name]['VALIDPERIOD'] = dataframes[df_name]['VALIDPERIOD'].astype(int)
    dataframes[df_name].fillna(1, inplace=True)


dataframes['user_list'].WITHDRAW_DATE.fillna(pd.Timestamp.max, inplace=True)
dataframes['user_list'].PREF_NAME.fillna(dataframes['user_list'].PREF_NAME.value_counts().index[0], inplace=True)

dataframes['user_list']['WITHDRAW_DATE'] = pd.to_datetime(dataframes['user_list']['WITHDRAW_DATE'])
dataframes['user_list']['REG_DATE'] = pd.to_datetime(dataframes['user_list']['REG_DATE'])

dataframes['coupon_detail_train'] = dataframes['coupon_detail_train'][['USER_ID_hash','COUPON_ID_hash','PURCHASEID_hash','I_DATE']]

for df_name in ['coupon_detail_train', 'user_list']:
    print(f'Saving to CPP_REPRO_{df_name}.csv...')
    dataframes[df_name].to_csv(f'CPP_REPRO_{df_name}.csv', index=False)

# Generate dataframe containing users and their purchased coupons
print('Creating training data...')
cvdict = dataframes['coupon_detail_train'][['USER_ID_hash','COUPON_ID_hash']].groupby('USER_ID_hash')['COUPON_ID_hash'].apply(list)
dataframes['user_list'] = pd.merge(dataframes['user_list'], dataframes['prefecture_locations'].drop('PREFECTUAL_OFFICE', axis=1), how='left')

# Generate positive and negative examples based on specifications in paper
examples_dict = {}

for coupon_type in ['positive', 'negative']:
    print(f'Creating {coupon_type} examples...')
    gen_function, target_val = (find_positive, 1) if coupon_type == 'positive' else (find_negative, 0)
    
    examples_dict[coupon_type] = dataframes['coupon_detail_train'].groupby('PURCHASEID_hash', group_keys=False).progress_apply(gen_function)
    examples_dict[coupon_type] = pd.merge(examples_dict[coupon_type], dataframes['prefecture_locations'].drop('PREFECTUAL_OFFICE', axis=1), how='left')
    examples_dict[coupon_type]['TARGET'] = target_val
    examples_dict[coupon_type] = pd.merge(examples_dict[coupon_type], dataframes['user_list'], how='left', on='USER_ID_hash', suffixes=('_COUPON', '_USER'))
    print(f'Saving to {coupon_type}_coupons_train.csv')
    examples_dict[coupon_type].to_csv(f'{coupon_type}_coupons_train.csv', index=False)

# examples_dict = {
#     'positive': pd.read_csv('positive_coupons_train.csv'),
#     'negative': pd.read_csv('negative_coupons_train.csv')
# }

print('Combining positive and negative examples...')
filter_columns = ['USER_ID_hash', 'COUPON_ID_hash', 'CAPSULE_TEXT', 'GENRE_NAME', 'PRICE_RATE', 'CATALOG_PRICE', 'DISCOUNT_PRICE', 'DISPFROM', 'DISPEND', 'DISPPERIOD', 'VALIDFROM', 'VALIDEND', 'VALIDPERIOD', 'USABLE_DATE_MON', 'USABLE_DATE_TUE','USABLE_DATE_WED', 'USABLE_DATE_THU', 'USABLE_DATE_FRI', 'USABLE_DATE_SAT', 'USABLE_DATE_SUN', 'USABLE_DATE_HOLIDAY', 'USABLE_DATE_BEFORE_HOLIDAY', 'LARGE_AREA_NAME', 'PREF_NAME_COUPON', 'SMALL_AREA_NAME', 'LATITUDE_COUPON', 'LONGITUDE_COUPON', 'REG_DATE', 'SEX_ID', 'AGE', 'WITHDRAW_DATE', 'PREF_NAME_USER', 'LATITUDE_USER', 'LONGITUDE_USER']
filter_columns_with_target = ['USER_ID_hash', 'COUPON_ID_hash', 'CAPSULE_TEXT', 'GENRE_NAME', 'PRICE_RATE', 'CATALOG_PRICE', 'DISCOUNT_PRICE', 'DISPFROM', 'DISPEND', 'DISPPERIOD', 'VALIDFROM', 'VALIDEND', 'VALIDPERIOD', 'USABLE_DATE_MON', 'USABLE_DATE_TUE','USABLE_DATE_WED', 'USABLE_DATE_THU', 'USABLE_DATE_FRI', 'USABLE_DATE_SAT', 'USABLE_DATE_SUN', 'USABLE_DATE_HOLIDAY', 'USABLE_DATE_BEFORE_HOLIDAY', 'LARGE_AREA_NAME', 'PREF_NAME_COUPON', 'SMALL_AREA_NAME', 'LATITUDE_COUPON', 'LONGITUDE_COUPON', 'REG_DATE', 'SEX_ID', 'AGE', 'WITHDRAW_DATE', 'PREF_NAME_USER', 'LATITUDE_USER', 'LONGITUDE_USER', 'TARGET']

dataset = pd.concat([examples_dict['positive'], examples_dict['negative']]).reset_index(drop=True)
dataset = dataset[filter_columns_with_target]
dataset.sort_values('TARGET', inplace=True)

print('Dropping false negative examples...')
dataset.drop_duplicates(subset=filter_columns, keep='first', inplace=True)

print('Saving to CPP_REPRO_coupon_list_train.csv...')
dataset.to_csv('CPP_REPRO_coupon_list_train.csv', index=False)

print('Creating test data...')
#Permutation of User-CouponTest
clist = dataframes['coupon_list_test'].COUPON_ID_hash.unique().tolist()
ulist = dataframes['user_list'].USER_ID_hash.unique().tolist()

relations = [r for r in itertools.product(clist, ulist)]
relations = pd.DataFrame(relations, columns=['COUPON_ID_hash','USER_ID_hash'])

dataframes['coupon_list_test'] = pd.merge(dataframes['coupon_list_test'], dataframes['prefecture_locations'].drop('PREFECTUAL_OFFICE', axis=1), how='left')
dataframes['coupon_list_test'] = pd.merge(relations, dataframes['coupon_list_test'], how='left')
dataframes['coupon_list_test'] = pd.merge(dataframes['coupon_list_test'], dataframes['user_list'], how='left', on='USER_ID_hash', suffixes=('_COUPON', '_USER'))
dataframes['coupon_list_test'] = dataframes['coupon_list_test'][filter_columns]
# dataframes['coupon_list_test'] = dataframes['coupon_list_test'][(dataframes['coupon_list_test'].DISPEND >= dataframes['coupon_list_test'].REG_DATE) & (dataframes['coupon_list_test'].DISPFROM <= dataframes['coupon_list_test'].WITHDRAW_DATE)]
print('Saving to CPP_REPRO_coupon_list_test.csv...')
dataframes['coupon_list_test'].to_csv('CPP_REPRO_coupon_list_test.csv', index=False)