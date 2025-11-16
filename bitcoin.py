import streamlit as st
import pandas as pd
import yfinance as yf
from pmdarima import auto_arima
import matplotlib.pyplot as plt
from datetime import timedelta
import numpy as np

# --- Configuration ---
TICKER_SYMBOL = 'BTC-USD'
FORECAST_DAYS = 14
DATA_PERIOD_HISTORY = '3y' # For stable ARIMA training (3 years of daily data)
DATA_PERIOD_LIVE = '1d'    # For fetching the most recent data
DATA_INTERVAL_LIVE = '1m'  # For fetching minute-level data

# --- Data Fetching and Model ---

@st.cache_data(ttl=60*60*4) # Cache for 4 hours
def load_historical_data(ticker):
    """Fetches long-term historical data for model training."""
    st.info(f"Fetching {DATA_PERIOD_HISTORY} of daily data for model training...")
    try:
        # Fetch long-term daily data for stable ARIMA training
        data = yf.download(ticker, period=DATA_PERIOD_HISTORY)
        df = data[['Close']].copy()
        df.dropna(inplace=True) 
        return df
    except Exception as e:
        st.error(f"Error fetching historical data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60) # Cache for 1 minute to simulate "live" refresh
def get_most_recent_price(ticker):
    """Fetches the most recent available price point (near-live)."""
    try:
        # Using Ticker.history for the most granular interval
        ticker_obj = yf.Ticker(ticker)
        # Fetch 1-minute data for the last 24 hours (1 day)
        recent_data = ticker_obj.history(period=DATA_PERIOD_LIVE, interval=DATA_INTERVAL_LIVE)
        
        if not recent_data.empty:
            # Get the very last closing price
            latest_price = recent_data['Close'].iloc[-1]
            latest_time = recent_data.index[-1]
            return latest_price, latest_time
        return None, None
    except Exception as e:
        # This warning is useful if the yfinance server times out
        st.warning(f"Could not retrieve most recent price: {e}")
        return None, None

@st.cache_resource(ttl=60*60*12) # Cache model training for 12 hours to speed up subsequent loads
def train_and_forecast(df, n_periods):
    """Trains the ARIMA model and generates a forecast."""
    if df.empty:
        return pd.DataFrame(), None

    with st.spinner('Training ARIMA model... This may take a moment on first run.'):
        try:
            # Auto_arima finds the best (p, d, q) parameters
            model = auto_arima(df['Close'], seasonal=False, stepwise=True,
                               suppress_warnings=True, error_action='ignore')
            
            # Forecast the next N periods
            forecast_values, conf_int = model.predict(n_periods=n_periods, return_conf_int=True, alpha=0.05)
            
            # Create forecast index (assuming daily forecast)
            last_date = df.index[-1]
            forecast_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                           periods=n_periods, freq='D')

            # Create the forecast DataFrame
            forecast_df = pd.DataFrame({
                'Predicted Price': forecast_values,
                'Lower Bound (95%)': conf_int[:, 0],
                'Upper Bound (95%)': conf_int[:, 1]
            }, index=forecast_dates)
            
            return forecast_df, model

        except Exception as e:
            st.error(f"Error during model training/forecasting: {e}")
            return pd.DataFrame(), None

# --- Streamlit App Layout ---
def main():
    st.set_page_config(page_title="₿ BTC 14-Day Price Predictor", layout="wide")
    st.title("₿ Bitcoin Price Prediction Project")
    st.markdown(f"Forecasting the next **{FORECAST_DAYS} days** of **{TICKER_SYMBOL}** using an **ARIMA** model.")

    # 1. Fetch and Display Most Recent Price
    latest_price, latest_time = get_most_recent_price(TICKER_SYMBOL)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("Current Price (Near-Live)")
        if latest_price is not None:
            st.metric(label=f"{TICKER_SYMBOL} Price", 
                      value=f"${latest_price:,.2f}", 
                      delta=f"As of {latest_time.strftime('%Y-%m-%d %H:%M %Z')}")
        else:
            st.warning("Could not fetch a recent price. Using historical data for prediction.")
            
    # 2. Load Historical Data and Train Model
    historical_df = load_historical_data(TICKER_SYMBOL)
    
    if historical_df.empty:
        st.error("Cannot proceed without historical data.")
        st.stop()
        
    with col2:
        st.success(f"Historical data loaded successfully! ({DATA_PERIOD_HISTORY} of data, ending {historical_df.index[-1].strftime('%Y-%m-%d')})")
        st.caption("Model is trained on daily data. The forecast is **daily** for 14 days.")

    # 3. Train Model and Forecast
    forecast_df, model = train_and_forecast(historical_df, FORECAST_DAYS)

    if forecast_df.empty:
        st.stop()
        
    st.markdown("---")

    # 4. Display Results

    ## Predicted Values Table
    st.header(f"🔮 {FORECAST_DAYS}-Day Predicted Prices")
    st.dataframe(
        forecast_df.style.format('${:,.2f}'), 
        use_container_width=True
    )
    st.caption("Forecast includes 95% confidence interval bounds. Prediction starts one day after the last historical close.")

    ## Combined Plot
    st.header("📈 Historical Data vs. 14-Day Forecast")

    # Plotting using Matplotlib
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot historical closing prices
    ax.plot(historical_df.index, historical_df['Close'], label='Historical Closing Price', color='blue', linewidth=1.5)
    
    # Plot the forecasted prices
    ax.plot(forecast_df.index, forecast_df['Predicted Price'], label='Predicted Price', color='red', linestyle='--', linewidth=2)
    
    # Plot confidence interval (shaded region)
    ax.fill_between(forecast_df.index, 
                    forecast_df['Lower Bound (95%)'], 
                    forecast_df['Upper Bound (95%)'], 
                    color='pink', alpha=0.5, label='95% Confidence Interval')

    ax.set_title(f'{TICKER_SYMBOL} Price Forecast', fontsize=18)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Price (USD)', fontsize=14)
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    st.pyplot(fig)
    
if __name__ == '__main__':
    main()
