import pandas as pd
import numpy as np
import holidays
import logging
import matplotlib.pyplot as plt
import seaborn as sns
def promo_dist(data):
    logging.info("counting the distriution of the data...")
    dist=data['Promo'].value_counts(normalize=True)
    return dist
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
def visualize_bef_af_hol_sales(data):#to see the sales during, before and after the holidays
    logging.info('plotting sales behavior:before, during and after holidays....')
    plt.figure(figsize=(12,6))
    sns.boxplot(x='period',y='Sales',data=data,order=['Before','During','After','Non-holiday'])
    plt.title('Sales Behavior: Before, during and after the Holidays')
    plt.xlabel('Period')
    plt.ylabel('Sales')
    plt.show()
def correlation_matrix(data):
    logging.info('visualizing correlation between sales and customers....')
    correlation=data['Sales'].corr(data['Customers'])
    return correlation
def promo_sales(data):
    logging.info('plotting sales behavior during a promotion....')
    plt.figure(figsize=(12,6))
    sns.boxplot(x='Promo',y='Sales',data=data, order= [1.0,0])
    plt.title('Sales Behavior: during a promotion')
    plt.xlabel('Promo')
    plt.ylabel('Sales')
    plt.show()
def sales_on_weekend(data):
    logging.info('visualizing correlation between sales and weekends....')
    data['Date'] = pd.to_datetime(data['Date'])

    # Extract weekday names (e.g., "Monday", "Tuesday")
    data['DayOfWeek'] = data['Date'].dt.day_name()
    stores_open_weekdays= data[data['DayOfWeek'].isin(['Monday','Tuesday','Wednseday','Thursday','Friday'])]
    stores_open_weekdays= stores_open_weekdays.groupby('Store').agg({'Open':'sum'})
    stores_open_weekdays=stores_open_weekdays[stores_open_weekdays['Open']==5]
    # Filter the stores that are open on all weekdays
    stores_open_all_weekdays = stores_open_weekdays.index
    # Now, analyze sales for these stores on weekends (Saturday and Sunday)
    weekend_sales = data[(data['Store'].isin(stores_open_all_weekdays)) & 
                   (data['DayOfWeek'].isin(['Saturday', 'Sunday']))]

    # You can calculate the average sales on weekends for stores open on all weekdays
    avg_weekend_sales = weekend_sales.groupby('Store')['Sales'].mean()

    # Plotting the weekend sales for these stores
    plt.figure(figsize=(10, 6))
    sns.barplot(x=avg_weekend_sales.index, y=avg_weekend_sales.values)
    plt.xticks(rotation=90)
    plt.xlabel('Store')
    plt.ylabel('Average Weekend Sales')
    plt.title('Average Weekend Sales for Stores Open on All Weekdays')
    plt.show()
def customer_open(data):
    logging.info('visualizing the relationship of customers during opening days...')
    plt.plot(data['Open'].astype(int), data['Customers'], marker='o', linestyle='-', color='b')  # x: 'Open', y: 'Customers'
    plt.title('Customer Behavior on Opening Days')
    plt.xlabel('Open (1 = Open, 0 = Closed)')
    plt.ylabel('Number of Customers')
    plt.grid(True)
    plt.show()
def comp_dist_effect(data):
    logging.info('visualizing the relationship of competition distance with sales...')
    bin=[0,4000,8000,12000,float('inf')]
    label = ['Close', 'Medium', 'Far', 'Very Far']
    data['Competition_dist_group'] = pd.cut(data['CompetitionDistance'],bins=bin,labels=label)
    group_sales=data.groupby('Competition_dist_group')['Sales'].agg(['mean','median','std','count'])
    plt.plot(data['CompetitionDistance'],)
    group_sales['count'].plot(kind='bar',color='skyblue',figsize=(10,6))
    plt.title('Average Sales by Competition Distance Group')
    plt.xlabel('Competition Distance Group')
    plt.ylabel('Average Sales')
    plt.xticks(rotation=0)
    plt.show()