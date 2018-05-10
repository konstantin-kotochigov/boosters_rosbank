# -*- coding: utf-8 -*-

import pandas
import os
import numpy
from collections import Counter

os.chdir("C:\\Users\\Konstantin\\git\\boosters\\boosters_rosbank\\data")

# Load data

train_raw = pandas.read_csv("train.csv", sep=",")
test_raw = pandas.read_csv("test.csv", sep=",")

## Task 1

train_raw['df'] = "train"
test_raw['target_flag'] =  numpy.nan
test_raw['target_sum'] =  numpy.nan
test_raw['df'] = "test"



df_raw = train_raw.append(test_raw)

df_raw.currency = df_raw.currency.astype(str)
df_raw.MCC = df_raw.MCC.astype(str)

# There are many categories in MCC and Currency variables so we limit to N most frequent

    num_mcc_top_categories = 20
    num_currency_top_categories = 20
    
    mcc_top_categories = [x[0] for x in Counter(df_raw.MCC).most_common(num_mcc_top_categories)]
    currency_top_categories = [x[0] for x in Counter(df_raw.currency).most_common(num_currency_top_categories)]
    
    df_raw['MCC'] = numpy.where(df_raw.MCC.isin(mcc_top_categories), df_raw.MCC, "other")
    df_raw['currency'] = numpy.where(df_raw.currency.isin(currency_top_categories), df_raw.currency, "other")



df_raw['dt'] = pandas.to_datetime(df_raw.TRDATETIME.str.slice(0,7), format="%d%b%y")
df_raw = df_raw.drop(labels=['target_sum'], axis=1)
df_raw = df_raw.drop(labels=["TRDATETIME"], axis=1)
df_raw = df_raw.drop(labels=["PERIOD"], axis=1)

# Cnvert categorical to dummy respresentation to compute all stats
df_dummies = pandas.get_dummies(df_raw[['MCC','channel_type','currency','trx_category']], prefix=['MCC','channel_type','currency','trx_category'])
df_dummies = pandas.concat([df_raw, df_dummies], axis=1)

mcc_columns = df_dummies.columns[df_dummies.columns.str.slice(0,12)=="MCC_"]
channel_columns = df_dummies.columns[df_dummies.columns.str.slice(0,13)=="channel_type_"]
currency_columns = df_dummies.columns[df_dummies.columns.str.slice(0,17)=="currency_"]
trx_category_columns = df_dummies.columns[df_dummies.columns.str.slice(0,13)=="trx_category_"]

df_dummies['cnt_dummy'] = 1

dummy_columns = list(mcc_columns.append(channel_columns).append(currency_columns).append(trx_category_columns)) + ['cnt_dummy']
not_dummy_columns = [x for x in df_dummies.columns if x not in dummy_columns]

df_dummies.sort_values(by=['cl_id','dt'], inplace=True)


# Generate features

    df_dummies.groupby(['cl_id']).apply(process_group).sort_values(by=['cl_id','dt']).head(100)
    
    def process_group(x):
        
        return (pandas.concat([x[not_dummy_columns], x[dummy_columns].cumsum()], axis=1))
    
    
    df_agg = df_dummies.groupby(['cl_id'])[dummy_columns].cumsum()


