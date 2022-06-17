# S&P500 Stock Price Forecasting

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
4. [Project Descriptions](#project)
4. [File Descriptions](#files)
5. [Results](#results)
6. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

    All libraries are available in Anaconda distribution of Python 3.*.
    Additional libraries:
        $ pip install yfinance
        $ pip install alpha_vantage
        $ pip install torch
        $ pip install statsmodels==0.12.1

Recommend to run the codes on [Google Colaboratory](https://research.google.com/colaboratory/)


## Project Motivation<a name="motivation"></a>

To help traders make decisions invest in stock

## Project Descriptions<a name="project"></a>

The project has two stages which are:

1. **Collecting and Preprocessing data:** `get_data.py` and `preprocess.py` 
contain the scripts to prepare data which:
    - Crawl data from Yahoo Finance and Alpha Vantage and merge the data
    - Filter Data by date range from Jan-01-2018 to Mar-15-2022
    - Fill in missing data
    - Drop useless factors
    - Store data to `.csv` files

2. **Training and Evaluating model:** The rest of the files contain the necessary script to train and evaluate the model which:
    - Build neural network model (RNN, LSTM, GRU)
    - Train and Evaluate Neural Network
    - Train traditional machine learning models
    - Train the ARIMA model
    - Evaluate all the models
    - Result Analysis

## File Descriptions <a name="files"></a>

The files structure is arranged as below:

	- README.md: read me file
    - \data
        - ...
    - \model
        - ...
    - get_data.py: Python Script to crawl data from Yahoo Finance and Alpha Vantage
    - preprocess.py: Cleaning the data
    - utilities.py: Self-defined function used in this project
    - rnn.py: RNN model
    - lstm.py: LSTM model
    - gru.py: GRU model
    - optimization.py: Train and Evaluate Neural Network
    - Linear Regression.ipynb: Train Linear Regression model 
    - Random Forest.ipynb: Train Random Forest model 
    - Ada Boost.ipynb: Train Ada Boost model 
    - Light Gradient Boosting Machine.ipynb : Train Light Gradient Boosting Machine model 
    - Extreme Gradient Boosting.ipynb: Train Extreme Gradient Boosting model 
    - ARIMA.ipynb: Train ARIMA model 
    - RNN.ipynb: Train RNN model 
    - LSTM.ipynb: Train LSTM model 
    - GRU.ipynb: Train GRU model 
    - Evaluation.ipynb: Evaluate all trained models
    - Result Analysis.ipynb: Get result stats

    
    
    
    


## Results<a name="results"></a>

|                           Model |      RMSE |       MAE |     MAPE |
|--------------------------------:|----------:|----------:|---------:|
|                       Ada Boost | 17.417070 | 14.603937 | 0.067849 |
| Light Gradient Boosting Machine | 17.924818 | 15.006538 | 0.069104 |
|       Extreme Gradient Boosting | 17.980285 | 15.103836 | 0.069642 |
|                   Random Forest | 18.149526 | 15.202881 | 0.069926 |
|                             RNN | 21.602305 | 19.228974 | 0.084792 |
|                            LSTM | 23.072414 | 20.607802 | 0.085224 |
|                             GRU | 22.288637 | 19.803434 | 0.086014 |
|                           ARIMA | 22.067823 | 18.670408 | 0.089866 |
|               Linear Regression | 20.695792 | 17.141832 | 0.197982 |

Ada Boost has the best results, which has Mean Absolute Percentage Error of only 6.785%

## Licensing, Authors, Acknowledgements<a name="licensing"></a>
The raw data used in this project belong to Yahoo Finance and Alpha Vantage