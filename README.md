# Stock Trend Prediction App


**Stock Trend Prediction App** is a web application that allows users to predict the future trends of a selected stock by analyzing its historical price data. This app utilizes deep learning techniques to provide insights into stock price movements.

## Features

- **Stock Data Analysis:** Fetches historical stock price data using the Yahoo Finance API.
- **Interactive Visualization:** Displays interactive charts of the stock's closing prices, moving averages, and predicted trends.
- **Deep Learning Prediction:** Utilizes a pre-trained deep learning model to predict the stock's future trend.
- **User-Friendly Interface:** Provides a user-friendly web interface for easy interaction.

## How to Use

Follow these steps to run the Stock Trend Prediction App on your local machine:

1. **Clone the Repository:**

   ```shell
   git clone https://github.com/https://github.com/Rafe2001/Stock_Trend_Prediction.git
   ```

2. **Navigate to the Project Directory:**

   ```shell
   cd stock-trend-prediction-app
   ```

3. **Install Dependencies:**

   ```shell
   pip install -r requirements.txt
   ```

4. **Run the App:**

   ```shell
   streamlit run main.py
   ```

5. **Open the App in Your Browser:**

   The app should open in your default web browser. If not, check the terminal for the app's URL.

6. **Enter a Stock Symbol:**

   Enter the stock symbol (e.g., AAPL for Apple Inc.) in the input field and press Enter.

7. **View Predictions and Charts:**

   - The app will display the stock's historical data, closing price plot, and moving averages.
   - You can also see the predicted trend for the next day: Up, Down, or Stable.

8. **Interact with the App:**

   Explore the app's features, analyze different stocks, and make predictions.

## Requirements

- Python 3.7 or higher
- Dependencies listed in `requirements.txt`
- An internet connection to fetch stock data

## Built With

- [Streamlit](https://streamlit.io/) - For building the web app interface
- [Yahoo Finance API](https://pypi.org/project/yfinance/) - For retrieving stock price data
- [TensorFlow/Keras](https://www.tensorflow.org/) - For deep learning model

## Author

- [Abdul Rafe Khan](https://github.com/Rafe2001)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to [Streamlit](https://streamlit.io/) for making it easy to create web apps with Python.
- The deep learning model used in this project is based on [Keras](https://www.tensorflow.org/guide/keras) and [TensorFlow](https://www.tensorflow.org/).
