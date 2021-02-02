---
toc: true
layout: post
title: Time Series  for beginners
categories: [timeseries]
---

If you are walking the path of becoming a data scientist, you might have already come across the term Time Series and you might have also realized the importance of Time Series Analysis and Forecating. In this post, I will try to give a gentle introduction so that it can kickstart your learning.

![]({{ site.baseurl }}/images/posts/2020-9-14/tss.png "time-series-plot")

## What is Time series?

As the name suggests, __time series__ is just a series of observations collected over different points in time. There exists a correlation between the observations collected at adjacent time points, therefore the previous observations of a variable can be used in predicting the same variable. This distinguishes time series data from general machine learning data where the observtions are collected at a single point in time.

The data collected over time represents a time series only if the observations are dependent on time. If the data collected is purely random in nature, forecating the future values is not possible and such data is called __white noise__. 

## Univariate vs Multivariate time series
If only a single variable is varying over time, it is called __Univariate time series__. For example, temperature of a room measured every hour. Here there are no other variables recorded, hence predicting temperature only depends on the temperature values recorded at previous time points.

If there are more than one variable varying over time, it is called __Multivariate time series__. For example, if the humidity is recorded along with the temperature then both temperature and humidity are to be considered in order to predict the temperature.

__Note__: Predecting the future is not the only goal of time series data. We can have different goals while working with time series. These goals can be mainly categorized into analysis and forecasting.

## Time Series Analysis
The main goal of time series analysis is exctracting useful statistics from data in order to understand the nature and underlying causes of the past. It helps to describe available data and provide interpretation to understand the problem domain better. Time series analyis can help to make better predictions.

## Time Series Forecasting
The main goal of forecasting is to build models on the past data and use them to predict future observations. For example, predicting number of births in a country based on the data collected in past years. This is challenging as the future observations are unavailable and must be predicted from what has already happened in the past.

## Stationarity in Time Series
If all the statistical characteristics of data like mean, auto correlation, variance do not vary with time then the time series is called stationary. But in general, most of the data recorded in not stationary i.e. the properties vary with time.

Analyzing such time series helps to understand the patterns such as trend, seasonality,cyclicality and irregularity. **Trend** is a general direction the data is changing as time passes. **Seasonality** is when a pattern recurs over fixed regular time intervals. **Cyclicality** is when there are any fluctuations around the trend. Unlike seasonality, cyclicality may vary in length. **Irregularity** is when there are random fluctuations which are not systematic and are irregular. These fluctuations cannot be controlled. These are called as time series components.

![]({{ site.baseurl }}/images/posts/2020-9-14/comps.jpg "Time series components(https://slideplayer.com/slide/8134442/)")

Most of the forecasting methods assume that the data is stationary because it is easy to predict the stationary data. Therfore, it is important to convert non-stationary data to stationary in order to apply forecasting models.

Check out [this](https://medium.com/data-science-in-your-pocket/why-time-series-has-to-be-stationary-37ca8800ddf) post by [Mehul Gupta](https://medium.com/@mehulgupta_7991) which explains Why time series has to be stationary more clearly.

It is important to analyze these components carefully in order to better understand the problem during analysis or forecasting. Since it is difficult to see all the components in a time series, a method called **Decomposition** can be used to identify them. These components can either combine in an *additive* way or in a *multiplicative* way.

An **Additive time series** is when the fluctuations in the data do not vary over time. Additive model is linear and seasonality has same frequency although the time increases.

> Time series  = trend + seasonality + cyclicality + irreguarity

A **Multiplicative time series** is when the variations or the fluctuations in the data increases as the time increases. Multiplicative model is non-linear and seasonality has either increasing or decreasing frequency.
> Time series = trend * seasonality * cyclicality * irregularity

## Time Series Decomposition
The purpose of decomposition is to identify and seperate components from a time series in order to perform better analysis and forecasting. 

In general, the cyclical component is hard to seperate and it is left by grouping it with the trend component, to form a trend-cycle component. It is often simply referred to as the trend component, even though it may contain cyclical behavior. 

Classical decomposition can be either a multiplicative or an additive decomposition. A function called [seasonal_decompose()](https://www.statsmodels.org/dev/generated/statsmodels.tsa.seasonal.seasonal_decompose.html) can be used to perform classical decomposition. You need to mention whether the model is additive or multiplicative.

Below is an example which shows decomposition of a dataset into components including the original data, trend, seasonality and irregularity(residual).

Let's first load the dataset and plot a simple graph:
```python
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('births.csv')
df.plot()
plt.show()
```
![]({{ site.baseurl }}/images/posts/2020-9-14/plot.png "data source(https://rb.gy/ehaaxw)")

Since the variations are very complex, we cannot see all the components clearly. Now, Decomposing this will give us clear picture of the components. Let's look how we can decompose this using **seasonal_decompose()** function:
```python
from statsmodels.tsa.seasonal import seasonal_decompose
components = seasonal_decompose(df['births'], model='multiplicative',period=10)
components.plot()
pyplot.show()
```
![]({{ site.baseurl }}/images/posts/2020-9-14/dec.png)

Now you can see all the components, you can analyze them and remove any of them them if not needed. For example, if you want to analyze the trend of a stock data, you would need to remove the seasonality found in the data and the noise due to irregularity.

Let's look into the basic steps to be followed while performing a forecasting task - 

## Basic steps for Forecasting

1. **Defining the problem**: Understanding the problem domain and clearly knowing the end goal of the forecast. The most important skill needed for a data scientist is being able to explain why a prediction is made and present results in a proper way. This is possible only with having a clear knowledge of who needs the forecast, why and how it will be used.
2. **Data Collection**: Collecting the past data related to the problem domain, gathering other important information from domain experts.
3. **Data preperation**: This includes exploring the data to know components like trend or seasonality, cleaning the data to fill the missing values and remove outliers if any, basic feature engineering to understand the relation ship between features or to add any new features, resampling and data transforms to remove noise and improve the forecasting.
4. **Modeling**: This includes configuring the right forecast model for the data. Widely used time series models are Auto Regressive(**AR**) models, Moving Average(**MA**) models, Intergrated(**I**) models and the combination of these models like Auto Regressive Moving Average models(**ARMA**), Auto Regressive Integrated Moving Average models(**ARIMA**). It is better to try models of different types, from simple to advanced approaches.
5. **Evaluation**: The time series forecasting model can only be trusted through its performance at predicting the future. This may include testing the model on previous data by creating train-test splits and calculating error or wait for the new observations to occur to compare the predictions.

## Applications of Time Series Forecasting
- Forecasting of agricultural commodity price
- Stock market analysis and forecasting
- Sales forecasting
- Forecasting supply chain components
- Weather forecasting
- Forecasting the birth rate in a country

any many more .....

## Conclusion

This is a basic introduction to time series for beginners. In this post, I've explained 
- What a time series is and why they are important.
- Components in a time series data.
- Decomposition of time series into it's components.
- Basic steps to be followed while performing a forecasting task.

*The data for plotting the graphs in this post is taken from [here](https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-total-female-births.csv).* 

---------
Thanks for reading! Please feel free to reach me via [twitter](https://twitter.com/jithendrabsy).
