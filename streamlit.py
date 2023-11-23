import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
# Set the aesthetic style of the plots
sns.set_style('whitegrid')

# We will create a pairplot to visualize the relationships between a subset of variables
# Selecting a subset of variables for the pairplot
variables = ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'cnt']

# Using tqdm to show progress
for column in tqdm(variables):
    df[column] = pd.to_numeric(df[column], errors='coerce')

# Dropping any rows with NaN values that may have resulted from coercion
pairplot_df = df.dropna(subset=variables)
# Creating the pairplot
sns.pairplot(pairplot_df[variables], hue='season', palette='viridis')
plt.show()
# Load the data
df = pd.read_csv('day.csv')
df['dteday'] = pd.to_datetime(df['dteday'])

# Sort the dataframe by date
sorted_df = df.sort_values('dteday')

# Streamlit title
st.title('Total Daily Bike Rentals Over Time')

# Streamlit line chart
st.line_chart(sorted_df.set_index('dteday')['cnt'])
# Since the previous attempt to create a pairplot did not produce an output,
# we will try again, ensuring that the data is processed correctly and efficiently.

# We will use a smaller subset of variables for the pairplot to ensure it can be processed
variables = ['season', 'temp', 'atemp', 'hum', 'windspeed', 'cnt']

# Convert 'season' to a categorical type for better plotting
pairplot_df['season'] = pairplot_df['season'].astype('category')

# Creating the pairplot with a smaller subset of variables
sns.pairplot(pairplot_df[variables], hue='season', palette='viridis', markers='+')
plt.show()
# Correcting the code to ensure the pairplot is displayed
# First, we ensure that 'season' is converted to a categorical type correctly

# Convert 'season' to a categorical type for better plotting
pairplot_df = df.dropna(subset=variables)
pairplot_df['season'] = pairplot_df['season'].astype('category')

# Now, we create the pairplot with the corrected dataframe
sns.pairplot(pairplot_df[variables], hue='season', palette='viridis', markers='+')
plt.show()
# Let's explore the usage patterns of the bike sharing system across different seasons.
# We will create a boxplot to compare the distribution of total bike rentals ('cnt') across different seasons.

# First, we convert 'season' to a more readable format
seasons = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}
pairplot_df['season_name'] = pairplot_df['season'].map(seasons)

# Now, we create the boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='season_name', y='cnt', data=pairplot_df, palette='coolwarm')
plt.title('Bike Rentals by Season')
plt.xlabel('Season')
plt.ylabel('Total Bike Rentals')
plt.show()
# To further understand the data, we will perform a time series decomposition
# to observe the trend, seasonality, and residuals of the bike rentals over time.
from statsmodels.tsa.seasonal import seasonal_decompose

# We need to ensure our data is in a time series format, with the date as the index
# and the total bike rentals ('cnt') as the values.
time_series_df = pairplot_df.set_index('dteday')['cnt']

# The frequency is daily, so we will set the model's frequency to 1
decomposition = seasonal_decompose(time_series_df, model='additive', period=1)

# Plotting the decomposed time series components
trend = decomposition.trend
trend.name = 'Trend'
seasonal = decomposition.seasonal
seasonal.name = 'Seasonality'
residual = decomposition.resid
residual.name = 'Residuals'

# Plotting all the components
plt.figure(figsize=(14, 8))
plt.subplot(411)
plt.plot(time_series_df, label='Original')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# Copyright notice
st.markdown('Copyright ARIS SUTIONO 2023')
