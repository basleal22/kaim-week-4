import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import holidays
import logging
def extract_data(data):
    data=pd.read_csv(data)
    return data
def extract_weekdays(data):
    # 1. Weekdays: Extract weekdays (0=Monday, 6=Sunday)
    data['Weekday'] = data['Date'].dt.weekday

    # 2. Weekends: Identify weekends (True if Saturday or Sunday)
    data['Weekend'] = data['Weekday'].isin([5, 6])
def add_holiday_col(data):
    logging.info("adding holiday period column ")
    data.index = pd.to_datetime(data.index)
    us_holidays=holidays.US()
    data['Is_holiday']= data.index.to_series().apply(lambda date: date in us_holidays).astype(int)
    data['period']= data.index.to_series().apply(lambda date: 
                                                 ('During' if date in us_holidays else
                                                  'Before' if (date-pd.Timedelta(days=7)) in us_holidays else
                                                  'After' if (date+pd.Timedelta(days=7)) in us_holidays else
                                                  'Non-holiday'
                                                  )
                                                  ) 
    return data
def days_to_holidays(data):    
    data.index = pd.to_datetime(data.index)
        #Get the US holidays
    us_holidays = holidays.US()
        #Number of Days to Holidays
    data['Days_to_Holiday'] = data.index.to_series().apply(
    lambda x: min((holiday - x).days for holiday in us_holidays) if len(us_holidays) > 0 else np.nan
    )
    return data
def days_aft_holidays(data):
    data.index = pd.to_datetime(data.index)
        #Get the US holidays
    us_holidays = holidays.US()
        # Calculate the number of days after the next holiday
    data['Days_After_Holiday'] = data.index.to_series().apply(
    lambda x: min((holiday - x).days for holiday in us_holidays if holiday > x) if any(holiday > x for holiday in us_holidays) else np.nan)
    return data
