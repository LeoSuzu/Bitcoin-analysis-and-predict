# Bitcoin Analysis

## Stock Price Analysis (2019-2023)

![Bitcoin Image](assets/pic/bitcoin-7260_256.gif)

### Author: Leo Suzuki / 2024

This notebook analyzes the Bitcoin stock prices from 2019 to 2023. Data is sourced from Yahoo Finance and Google Trends.

- **Yahoo Finance Data**: [Bitcoin Stock Price](https://finance.yahoo.com/quote/BTC-USD/history?p=BTC-USD)
- **Google Trends Data**: [Bitcoin Search Trends](https://trends.google.com/trends/explore?date=2023-01-01%202023-12-31&q=bitcoin&hl=en-US)

### Data Overview

- **Bitcoin Stock Data**: Bitcoin stock prices from January 2019 to December 2023.
- **Search Trend Data**: Google search trends for Bitcoin from January 2019 to December 2023.

---

## Libraries Used

The following libraries are used for data manipulation, visualization, and analysis:

- `pandas`
- `datetime`
- `numpy`
- `matplotlib`
- `plotly`
- `scikit-learn` (for data imputation)

---

## Data Preparation

1. **Reading Data**:
   - Bitcoin stock data is read from a CSV file (`BTC-USD_Jan2019_Dec2023.csv`).
   
2. **Data Examination**:
   - A quick look at the data's structure using `info()` and statistical summary using `describe()`.
   
3. **Handling Missing Values**:
   - Missing values are detected and handled (using `KNNImputer` if needed).

4. **Date Conversion**:
   - The `Date` column is converted to a datetime format for time-based operations.

5. **Renaming Columns**:
   - Columns are renamed for better readability: `['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']`.

---

## Visualizations and Analysis

### 1. **Yearly Analysis (2022 & 2023)**

- **Monthly Aggregation**:
  - Monthly average of open, high, low, and close prices for both 2022 and 2023.
  
- **Bar Charts**:
  - Bar charts are used to display monthly Bitcoin prices.
  
- **Daily Line Charts**:
  - Daily trends for open, high, low, and close prices are visualized using line charts.
  
- **Range Slider**:
  - Interactive range sliders are included to navigate through daily data.

### 2. **Five-Year Analysis (2019-2023)**

- **Average Price Calculation**:
  - The average price for each day is calculated by averaging the opening and closing prices.
  
- **Line Charts (Yearly Comparison)**:
  - Each year’s average stock price is displayed using different colors for comparison.

- **Special Event Annotation**:
  - The chart highlights significant dates, such as Bitcoin halving (May 11, 2020).

### 3. **Search Trend Data (Google Trends)**

- **Data Overview**:
  - Bitcoin search trend data from 2019 to 2023 is combined with the stock data.

- **Line Chart**:
  - The search trend data is plotted against Bitcoin’s average price to observe any correlations.

### 4. **Stock Volume vs Search Trend**

- **Correlation**:
  - A dual-axis chart shows the comparison between Bitcoin stock volume and search trends.
  
- **Date Formatting**:
  - Custom date tickers are applied to improve readability.

---

## Special Notes

- **Imputation**:
  - Missing values in the data are imputed using the `KNNImputer` from `scikit-learn` if necessary.
  
- **Data Sources**:
  - Bitcoin stock data is sourced from Yahoo Finance.
  - Bitcoin search trend data is sourced from Google Trends.
  
---

## Conclusion

This notebook provides an in-depth analysis of Bitcoin’s stock price from 2019 to 2023. Through various charts and visualizations, key trends, patterns, and correlations between stock price and search trends are explored.

### Future Work:
- Incorporate more advanced machine learning models to predict future Bitcoin trends based on stock data and search trends.


# Bitcoin Stock Price Prediction using LSTM and CNN Models

## Overview

![Stock Market Image](assets/pic/stock-market.jpg)

### Author: Your Name / 2024

This project involves predicting stock prices for the next 30 days using Long Short-Term Memory (LSTM) and Convolutional Neural Networks (CNN). The dataset consists of historical stock prices, and the models are evaluated based on performance metrics and prediction accuracy.

- **Data Source**: [Stock Price Data](https://example.com/stock-price-data)

## Data Overview

- **Historical Stock Prices**: Data includes daily stock prices with features such as adjusted closing prices.

## Libraries Used

The following libraries are used for data manipulation, visualization, and model training:

- `numpy`
- `pandas`
- `scikit-learn`
- `tensorflow`
- `keras`
- `plotly`
- `matplotlib`

## Model Training

### LSTM Model

#### Model Architecture
- **LSTM Layers**: Utilizes Long Short-Term Memory layers with dropout for regularization.
- **Dense Layers**: Fully connected layers for final output prediction.

#### Training Process
- **Epochs**: Specified number of epochs for training.
- **Batch Size**: Defined batch size for training.
- **Optimizer**: Adam optimizer with a lower learning rate.

#### Evaluation Metrics
- **RMSE**: Root Mean Squared Error
- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error

### CNN Model

#### Model Architecture
- **Conv1D Layers**: Convolutional layers followed by MaxPooling layers.
- **Dropout Layers**: Added dropout for regularization.
- **Dense Layers**: Fully connected layers for final output prediction.

#### Training Process
- **Epochs**: Specified number of epochs for training.
- **Batch Size**: Defined batch size for training.
- **Optimizer**: Adam optimizer with a lower learning rate.

#### Evaluation Metrics
- **RMSE**: Root Mean Squared Error
- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error

## Predictions

### LSTM Model Predictions

- **Next 30 Days**: Predicted stock prices for the next 30 days using the LSTM model.
- **Plot**: Visualization of predicted stock prices compared to historical data.

### CNN Model Predictions

- **Next 30 Days**: Predicted stock prices for the next 30 days using the CNN model.
- **Plot**: Visualization of predicted stock prices compared to historical data.

## Performance Comparison

### LSTM vs CNN

- **LSTM Model Metrics**: RMSE, MSE, and MAE for training and test datasets.
- **CNN Model Metrics**: RMSE, MSE, and MAE for training and test datasets.

### Visualizations

- **Stock Prices with Predictions**: Comparison plots of historical stock prices and model predictions for both LSTM and CNN.

## Conclusion

This project provides insights into predicting stock prices using advanced neural network models and evaluates their performance through various metrics and visualizations.

