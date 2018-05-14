# -*- coding: utf-8 -*-

import pandas
import os
import numpy
import math
from pandas import DataFrame, Series
from collections import Counter

os.chdir("/home/kkotochigov/")


# Load data

train_raw = pandas.read_csv("train.csv", sep=",")
test_raw = pandas.read_csv("test.csv", sep=",")


# Feature generation for both datasets

train_raw['df'] = "train"
test_raw['df'] = "test"
    
test_raw['target_flag'] =  numpy.nan
test_raw['target_sum'] =  numpy.nan
    
df_raw = train_raw.append(test_raw)


# Type conversion
    
df_raw.currency = df_raw.currency.astype(str)
df_raw.MCC = df_raw.MCC.astype(str)

# Currency conversion

approximate_currency_rate = df[['currency','amount']].groupby(['currency']).mean() / df.amount[df.currency=="810"].mean()
approximate_currency_rate = approximate_currency_rate.amount
df_raw['amount'] = df_raw.amount / df_raw.currency.map(approximate_currency_rate)


# There are many categories in MCC and Currency variables so we limit to N most frequent

num_mcc_top_categories = 20
num_currency_top_categories = 20
    
mcc_top_categories = [x[0] for x in Counter(df_raw.MCC).most_common(num_mcc_top_categories)]
currency_top_categories = [x[0] for x in Counter(df_raw.currency).most_common(num_currency_top_categories)]
    
df_raw['MCC'] = numpy.where(df_raw.MCC.isin(mcc_top_categories), df_raw.MCC, "other")
df_raw['currency'] = numpy.where(df_raw.currency.isin(currency_top_categories), df_raw.currency, "other")


# Set Date-based columns

df_raw['dt'] = pandas.to_datetime(df_raw.TRDATETIME.str.slice(0,7), format="%d%b%y")
df_raw['dow'] = df_raw['dt'].dt.dayofweek.astype(str)
    
min_trn_dt = df_raw['dt'].min()
df_raw['trx_week'] = ((df_raw['dt'] - min_trn_dt).dt.days / 7).map(math.floor)


# Delete unused

df_raw = df_raw.drop(labels=['target_sum'], axis=1)
df_raw = df_raw.drop(labels=["TRDATETIME"], axis=1)
df_raw = df_raw.drop(labels=["PERIOD"], axis=1)



# Cnvert categorical to dummy respresentation to compute all stats

df_dummies = pandas.get_dummies(df_raw[['MCC','channel_type','currency','trx_category','dow']], prefix=['MCC','channel_type','currency','trx_category','dow'])
df_dummies['cnt'] = 1
df = pandas.concat([df_raw, df_dummies], axis=1)

# Set variable lists    

mcc_columns = df_dummies.columns[df_dummies.columns.str.slice(0,4)=="MCC_"]
channel_columns = df_dummies.columns[df_dummies.columns.str.slice(0,13)=="channel_type_"]
currency_columns = df_dummies.columns[df_dummies.columns.str.slice(0,9)=="currency_"]
trx_category_columns = df_dummies.columns[df_dummies.columns.str.slice(0,13)=="trx_category_"]
dow_columns = df_dummies.columns[df_dummies.columns.str.slice(0,4)=="dow_"]

dummy_columns = list(mcc_columns.append(channel_columns).append(currency_columns).append(trx_category_columns).append(dow_columns)) + ['cnt']
not_dummy_columns = [x for x in df.columns if x not in dummy_columns]




# Generate features

df.sort_values(by=['cl_id','dt'], inplace=True)
# df.groupby(['cl_id']).apply(process_group).sort_values(by=['cl_id','dt']).head(100)
    
# PreAggregate Weekly
    
df_weekly = df[['cl_id','trx_week'] + dummy_columns].groupby(['cl_id','trx_week'], as_index=False).sum()
df_customers = DataFrame({"cl_id":df['cl_id'].unique(),"key":1})
df_weeks = DataFrame({"trx_week":list(range(0,df['trx_week'].max())), "key":1})
df_weekly_base = pandas.merge(df_customers, df_weeks, on=["key"])
df_weekly_base.drop(['key'],axis=1)
df_weekly_agg = pandas.merge(df_weekly_base, df_weekly, on=['cl_id','trx_week'], how='left', indicator=True)
df_weekly_agg.fillna(0)
# df_weekly_agg.columns
# df_weekly_agg.iloc[:,[0,1,2,3,4,18,19]].head(50)
# dummy_columns
    
# df_weekly_base.head(100)
    
def process_group(x):
        
    return (pandas.concat([x[not_dummy_columns], x[dummy_columns].cumsum()], axis=1))
    
    
# df_agg = df_dummies.groupby(['cl_id'])[dummy_columns].cumsum()
    
# df_dummies['week_number'] = df_dummies['dt'].dt.year.map(str) + df_dummies['dt'].dt.week.map(str)
   
df_weekly_agg[['cl_id','trx_week','cnt','dow_0','dow_1','dow_2','dow_3','dow_4','dow_5','dow_6']][df_weekly_agg.trx_week > 30]