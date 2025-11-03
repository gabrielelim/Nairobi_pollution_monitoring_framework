import streamlit as st
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
import seaborn as sns
import matplotlib.pyplot as plt
import datetime



st.title("🎈 NAIROBI SHORT LIVED CLIMATE POLLUTANTS MONITORING DASHBOARD")
st.write(
    "ADVOCATING AIR QUALITY FORECASTING THROUGH OPEN DATA TOOLS AND MACHINE LEARNING TECHNOLOGY."

)

# --- API Configuration ---
url = "https://air-quality-api.open-meteo.com/v1/air-quality"
today = datetime.date.today()
yesterday = today - datetime.timedelta(days=1)
params = {
    "latitude": -1.45,
    "longitude": 36.66,
    "hourly": ["ozone", "pm2_5",'nitrogen_dioxide'],
    "start_date": "2025-01-01",
    "end_date": yesterday.strftime("%Y-%m-%d"),
    "timezone": "Africa/Nairobi"
}

# --- Caching and Retries for API Calls ---
cache_session = requests_cache.CachedSession('.cache', expires_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Use Streamlit's cache to store the API response
@st.cache_data(ttl=86400) # Data will be re-fetched after 1 hour
def load_data(url, params):
    """
    Loads data from the Open-Meteo API.
    Returns a list of WeatherAPIResponse objects.
    """
    try:
        responses = openmeteo.weather_api(url, params=params)
        return responses
    except Exception as e:
        st.error(f"Error fetching data from the API: {e}")
        return None

# Attempt to load the data
responses = load_data(url, params)

# Check if responses were successfully fetched before processing
if responses:
    all_hourly_data = []
    
    # The API returns a list of responses, we need to process each one
    for response in responses:
        hourly = response.Hourly()
        hourly_ozone = hourly.Variables(0).ValuesAsNumpy()
        hourly_pm = hourly.Variables(1).ValuesAsNumpy()
        hourly_no = hourly.Variables(2).ValuesAsNumpy()

        # Create a pandas DataFrame for the current response
        hourly_data = {
            "date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit='s', utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive='left'
            ),
            'ozone': hourly_ozone,
            'pm': hourly_pm,
            'no':hourly_no
        }
        
        # Add the DataFrame to a list
        df = pd.DataFrame(data=hourly_data)
        df['date'] = pd.to_datetime(df['date'], utc=True)
        df['date'] = pd.to_datetime(df['date']).dt.tz_convert('Africa/Nairobi')
        df['day'] = df['date'].dt.date          # Extract the day
        df.set_index('date', inplace=True)      # Set the index to the datetime column

        
        all_hourly_data.append(df)  

        
    # Concatenate all dataframes if there are multiple responses
    if all_hourly_data:
        combined_df = pd.concat(all_hourly_data)
        st.dataframe(combined_df)
        sunheader = st.subheader("Hourly Ozone and PM2.5 Levels")
    else:
        st.info("No hourly data found in the API response.")
else:
    st.info("No data could be loaded from the API. Please check your internet connection or the API URL.")

# --- Ensure combined_df exists and is well-formed ---
if 'combined_df' in locals() and not combined_df.empty:
    # make sure index is datetime and timezone-aware
    if not isinstance(combined_df.index, pd.DatetimeIndex):
        combined_df.index = pd.to_datetime(combined_df.index)
    # ensure 'day' column exists (date part in local TZ)
    if 'day' not in combined_df.columns:
        combined_df['day'] = combined_df.index.date

    # --- Rolling / aggregated columns ---
    # 8-hour rolling mean for ozone (hourly)
    if 'ozone' in combined_df.columns:
        combined_df['ozone_8hr_rolling'] = combined_df['ozone'].rolling(window=8, min_periods=1).mean()
    else:
        combined_df['ozone_8hr_rolling'] = pd.NA

    # 24-hour rolling max for NO2 (hourly)
    if 'no' in combined_df.columns:
        combined_df['no2_24hr_rolling_max'] = combined_df['no'].rolling(window=24, min_periods=1).max()
    else:
        combined_df['no2_24hr_rolling_max'] = pd.NA

    # --- Daily aggregation ---
    df_daily = combined_df.groupby('day').agg(
        daily_ozone_mean=('ozone', 'mean'),
        daily_pm_average=('pm', 'mean'),
        ozone_8hr_rolling_daily_mean=('ozone_8hr_rolling', 'mean'),
        daily_no2_max=('no', 'max'),
        no2_24hr_rolling_daily_mean=('no2_24hr_rolling_max', 'mean')
    ).reset_index()

    # --- Hourly Ozone plot (raw + 8-hr rolling) ---
    if 'ozone' in combined_df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.set_style("whitegrid")
        ax.plot(combined_df.index, combined_df['ozone'], label='Hourly ozone', color='tab:blue', alpha=0.6, zorder=1)
        if combined_df['ozone_8hr_rolling'].notna().any():
            ax.plot(combined_df.index, combined_df['ozone_8hr_rolling'], label='8-hr rolling mean', color='tab:green', linewidth=2, zorder=2)
            threshold_hourly_o3 = 100
            above_hourly = combined_df[combined_df['ozone_8hr_rolling'] > threshold_hourly_o3]
            if not above_hourly.empty:
                ax.scatter(above_hourly.index, above_hourly['ozone_8hr_rolling'], color='red', s=50,
                           label=f'8-hr mean > {threshold_hourly_o3} µg/m³', zorder=5)
            ax.axhline(y=threshold_hourly_o3, color='orange', linestyle='--',
                       label=f'Guideline {threshold_hourly_o3} µg/m³', zorder=0)
        ax.set_title('Hourly Ozone and 8-hour Rolling Mean (Nairobi)')
        ax.set_xlabel('Datetime')
        ax.set_ylabel('Ozone (µg/m³)')
        ax.legend()
        ax.grid(True)
        fig.autofmt_xdate()
        plt.tight_layout()
        st.pyplot(fig)

    # --- Hourly NO2 plot (raw + 24-hr rolling max) ---
    if 'no' in combined_df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.set_style("whitegrid")
        ax.plot(combined_df.index, combined_df['no'], label='Hourly NO2', color='tab:purple', alpha=0.6, zorder=1)
        if combined_df['no2_24hr_rolling_max'].notna().any():
            ax.plot(combined_df.index, combined_df['no2_24hr_rolling_max'], label='24-hr rolling max', color='tab:orange', linewidth=2, zorder=2)
            threshold_no2 = 40
            above_no2 = combined_df[combined_df['no2_24hr_rolling_max'] > threshold_no2]
            if not above_no2.empty:
                ax.scatter(above_no2.index, above_no2['no2_24hr_rolling_max'], color='red', s=50,
                           label=f'24-hr max > {threshold_no2} µg/m³', zorder=5)
            ax.axhline(y=threshold_no2, color='gray', linestyle='--', label=f'Guideline {threshold_no2} µg/m³', zorder=0)
        ax.set_title('Hourly NO2 and 24-hour Rolling Max (Nairobi)')
        ax.set_xlabel('Datetime')
        ax.set_ylabel('NO2 (µg/m³)')
        ax.legend()
        ax.grid(True)
        fig.autofmt_xdate()
        plt.tight_layout()
        st.pyplot(fig)

    # --- Daily PM2.5 plot ---
    if not df_daily.empty and 'daily_pm_average' in df_daily.columns:
        st.subheader(f"Daily average PM2.5 levels up to {yesterday.strftime('%Y-%m-%d')}")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.set_style("whitegrid")
        threshold_pm = 15
        # plot hourly PM2.5 from combined_df (was incorrectly using df_hourly)
        if 'pm' in combined_df.columns:
            ax.plot(combined_df.index, combined_df['pm'], label='Hourly PM2.5', color='tab:blue', alpha=0.6, zorder=1)
        sns.lineplot(x='day', y='daily_pm_average', data=df_daily, marker='o', label='Daily PM2.5 average', ax=ax, zorder=2)
        above_threshold_pm = df_daily[df_daily['daily_pm_average'] > threshold_pm]
        if not above_threshold_pm.empty:
            sns.scatterplot(x='day', y='daily_pm_average', data=above_threshold_pm, color='red', s=100,
                            label=f'PM2.5 > {threshold_pm} µg/m³', ax=ax, zorder=3)
        ax.axhline(y=threshold_pm, color='orange', linestyle='--', label=f'WHO Guideline (Daily Mean {threshold_pm} µg/m³)', zorder=0)
        ax.set_title('Daily Average PM2.5 Concentration in Nairobi')
        ax.set_xlabel('Date')
        ax.set_ylabel('PM2.5 Concentration (µg/m³)')
        ax.legend()
        ax.grid(True)
        fig.autofmt_xdate()
        plt.tight_layout()
        st.pyplot(fig)

    

    # --- Daily maximum NO2 plot ---
    if not df_daily.empty and 'daily_no2_max' in df_daily.columns:
        st.subheader(f"Daily maximum NO2 levels up to {yesterday.strftime('%Y-%m-%d')}")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.set_style("whitegrid")
        sns.lineplot(x='day', y='daily_no2_max', data=df_daily, marker='o', label='Daily NO2 max', ax=ax, zorder=1)
        threshold_no2 = 20
        above = df_daily[df_daily['daily_no2_max'] > threshold_no2]
        if not above.empty:
            sns.scatterplot(x='day', y='daily_no2_max', data=above, color='red', s=100,
                            label=f'NO2 > {threshold_no2} µg/m³', ax=ax, zorder=2)
        ax.axhline(y=threshold_no2, color='orange', linestyle='--', label=f'Guideline {threshold_no2} µg/m³', zorder=0)
        ax.set_title('Daily Maximum NO2 Concentration in Nairobi')
        ax.set_xlabel('Date')
        ax.set_ylabel('NO2 (µg/m³)')
        ax.grid(True)
        fig.autofmt_xdate()
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("No daily NO2 data available to plot.")
else:
    st.info("No combined hourly data available to compute plots.")



