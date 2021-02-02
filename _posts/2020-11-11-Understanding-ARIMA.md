---
toc: true
layout: post
title: Breaking down - ARIMA
tags: [timeseries]
---

In a lot of machine learning problems the thing we want to predict is dependent on very clear inputs, such as properties of the target, pixels of an image, etc. In time series these indepent variables are often not known or does not exist.

For example, in predicting the demand of fresh vegetable products from a market, we don’t have a clear independent set of variables where we can fit a model on. 
A market collects vegetables from various farmers of different places. Is demand of a vegtable product dependent on the properties of a farm it was grown in, 
or the temperature at that place or the height and weight of the farmer? No, the demand of a vegetable does not depend on the farm or the farmer. The independent data is not easily available or even 
if we can try to find a relationship between such independent variables and the vegetable demand, these relationships are not perfect and clear. 

The **time series analysis** is frequently used in such cases. The fundamental intuition behind time series forecasting is that, the measure of some variable 
at a time period will depend on the measure of the same variable at previous time periods. Therefore, we analyze the series on the series itself.

![]({{ site.baseurl }}/images/posts/2020-11-11/i.jpeg)


One of the most used models when handling time series are ARIMA models. ARIMA stands for Autoregressive Integrated Moving Average. 
I will walk through all the parts of ARIMA to fully explain them.

## AR: Auto Regressive model
Auto Regressive (AR) model is a specific type of regression model where, the dependent variable depends on past values of itself.

For example, consider a stock price data. The stock price of today is influenced by yesterday's price. If the price(close price) of day t-1 is x dollars, the price of day t starts with x dollars. We can assume the price can be determined by the following model:

<i>Y<sub>t</sub> = μ + ϕY<sub>t-1</sub> + ϵ<sub>t</sub></i>

where μ  and ϕ are constants, and ϵ<sub>t</sub> is white noise.

This model is called AR model, and generally AR(p) is given by the following definition:

<i>Y<sub>t</sub> = μ + ϵ<sub>t</sub> + ϕ<sub>1</sub>Y<sub>t-1</sub> + ϕ<sub>2</sub>Y<sub>t-2</sub> + ...+ ϕ<sub>p</sub>Y<sub>t-p</sub></i>

## MA: Moving Average model
Moving Average (MA) model works by analysing how wrong you were in predicting values for the previous time-periods to make a better estimate for the current time-period. This model factors in errors from the observations. 

Generally MA(q) is given by the following definition:

<i>Y<sub>t</sub> = μ + ϵ<sub>t</sub> + θ<sub>1</sub> ϵ<sub>t-1</sub> + θ<sub>2</sub> ϵ<sub>t-2</sub>  + ...+ θ<sub>q</sub> ϵ<sub>t-q</sub></i>

MA model complements the AR model by taking the errors from the previous time-periods  into considerations, to get better forecast results.

The combined model between AR and MA is called ARMA model, and it is defined as AR(p,q). It is given as follows:

<i>Y<sub>t</sub> = μ + ϵ<sub>t</sub> + ϕ<sub>1</sub>Y<sub>t-1</sub> + ...+ ϕ<sub>p</sub>Y<sub>t-p</sub> + θ<sub>1</sub> ϵ<sub>t-1</sub> + ...+ θ<sub>q</sub> ϵ<sub>t-q</sub></i>

## Estimating the ARMA(p,q) model hyperparameters

When selecting the parameters p and q, we must focus on characteristics of the model. Each model have different characteristics for autocorrelation function (ACF) and partial autocorrelation function (PACF). By looking at the ACF and PACF plots, we can identify the numbers of MA and AR terms i.e. q and p respectively.

### Autocorrelation
> The relationship between two variables is summarized by correlation.

If the correlation is calculated for the timeseries observations with the observations at previous time steps is called as **Autocorrelation**. The observations of previous timesteps are called as lags. The __ACF plot__ is a bar chart of the coefficients of correlation between timeseries and lags of itself. 

### Partial Autocorrelation
> The "partial" correlation between two variables is the amount of correlation between them which is not explained by their mutual correlations with a specified set of other variables.

If the correlation is calculated for the timeseries observations with the observations at previous time steps at lag k by eliminating all the effects of shorter lags i.e. 1,2,...k-1, then it is called as __Partial Autocorrelation__. The __PACF plot__ is a plot of the partial correlation coefficients between the series and lags of itself.

## Characteristics of ACF and PACF
Now, the models have the following characteristics for the autocorrelation and partial autocorrelation.

- AR(p): When the lag is getting large, The autocorrelation decreases exponentially and the partial autocorrelation cuts-off after lag p.
- MA(q): When the lag is getting large, the autocorrelation cuts-off after lag q and the partial autocorelation tails off to zero.
- ARMA(p,q): When the lag is getting large, both autocorrelation and partial autocorrelation tails off to zero.

Using these characteristics, we can estimate the proper model.

## Stationarity
AR, MA and ARMA models require the data to be stationary. A stationary series has a constant mean and variance over time. But in real-world, most of the data is not stationary. 

So how to forecast non-stationary series?

Well, here comes the ARIMA model which works with data that is not stationary. The new term added for ARIMA is I.

## Integrated (I)
Consider a non-stationary series which needs to be forecasted. We can say whether the data is stationary or not by studying the plot against time. 

![]({{ site.baseurl }}/images/posts/2020-11-11/1.png "Original time series plot (Non-stationary)")

It is clearly visible that the mean is increasing over time i.e. the series is not stationary. If this upward trend is eliminated, the series becomes stationary. The easiest way to do this is to consoder the differences between consecutive timesteps. It goes as follows:

<i>I<sub>t</sub> = Y<sub>t+1</sub> - Y<sub>t</sub></i>

After applying this transformation, the series becomes like this with observable linear trend - 

![]({{ site.baseurl }}/images/posts/2020-11-11/2.png "Differenced time series plot (stationary)")

The transformed series is called as the differenced series. This differenced series is used for forecasting instead of the timeseries. This step of converting non-stationary series to stationary series results in the new term I which stands for Integrated. (Note: This has nothing to with integration).

In the above example, the data becomes stationary after performing the first order differencing which means a single time differencing of observations at consecutive timesteps. But in some cases, the data remains non-stationary after performing the first order differencing. Hence, the series could be differenced more than once to make it stationary.

d = 1 : 
<i>I<sub>t</sub> = Y<sub>t+1</sub> - Y<sub>t</sub></i>

d = 2 : <i>J<sub>t</sub> = I<sub>t+1</sub> - I<sub>t</sub>  =  Y<sub>t+2</sub> + Y<sub>t</sub> - 2Y<sub>t+1</sub></i>

Hence, ARIMA has hyperparameters p, q and d:
- p - order of the AR model
- q - order of the MA model
- d - order of differencing

## Seasonality
It is to be noted that ARIMA model assumes that the data is not seasonal. The presence of variations that occur at specific regular intervals, such as weekly or monthly results in seasonality of time series. For example, Arrival of vegetables to the market or sales of Diwali crackers. __ARIMA does not work well for seasonal series__.

There is an upgrade of ARIMA model, called __Seasonal ARIMA__ or __SARIMA__. It is represented as ARIMA (p,d,q) X (P,D,Q,m).
- p - trend AR order
- q - trend MA order
- d - trend difference order
- P - seasonal AR order
- Q - seasonal MA order
- D - seasonal difference order
- m - frequency of seasonality in timeseries

To identify values for the seasonal model, ACF and PACF plots can be analyzed by looking at correlation at seasonal lag time steps i.e. the lag at which seasonality occurs. Alternately, [this post](https://machinelearningmastery.com/how-to-grid-search-sarima-model-hyperparameters-for-time-series-forecasting-in-python/) explains how grid searching can be used across the trend and seasonal hyperparameters.

Another easy way to use is to use [pmdarima](http://alkaline-ml.com/pmdarima/)'s auto.arima which automatically gives the best fit model.

## Further Reading
- [Fundamentals of Time series Data and Analysis](https://www.aptech.com/blog/introduction-to-the-fundamentals-of-time-series-data-and-analysis/)
- [ARIMA models for time series forecasting](http://people.duke.edu/~rnau/411arim.htm)
- [Summary of rules for identifying ARIMA models](http://people.duke.edu/~rnau/arimrule.htm)
- [Notes on nonseasonal ARIMA models](http://people.duke.edu/~rnau/Notes_on_nonseasonal_ARIMA_models--Robert_Nau.pdf)
- [How does auto.arima() work?](https://otexts.com/fpp2/arima-r.html)

## Credits
Thanks for reading! And please feel free to connect to me via [twitter](https://twitter.com/jithendrabsy) or [linkedin](https://www.linkedin.com/in/jithendrabsy/). Feedback is always welcome.
