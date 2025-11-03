import streamlit as st
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
import seaborn as sns
import matplotlib.pyplot as plt
import datetime



st.title("🎈 NAIROBI CLEAN AIR ADVOCACY")
st.write(
    "ADVOCATING AIR QUALITY THROUGH OPEN DATA TOOLS AND MACHINE LEARNING TECHNOLOGY."

)

# --- API Configuration ---
url = "https://air-quality-api.open-meteo.com/v1/air-quality"
today = datetime.date.today()
yesterday = today - datetime.timedelta(days=1)
params = {
    "latitude": -1.45,
    "longitude": 36.66,
    "hourly": ["ozone", "pm2_5"],
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

        # Create a pandas DataFrame for the current response
        hourly_data = {
            "date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit='s', utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive='left'
            ),
            'ozone': hourly_ozone,
            'pm': hourly_pm
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


####

#
combined_df['ozone_8hr_rolling']= combined_df['ozone'].rolling(window=8,min_periods=1).mean()
df_daily = combined_df.groupby('day').agg(daily_ozone_mean=('ozone','mean'),daily_pm_average = ('pm','mean'),
                                          ozone_8hr_rolling_daily_mean=('ozone_8hr_rolling','mean')).reset_index()
#st.dataframe(df_daily)
#############
st.subheader(f"Daily average PM2.5 levels upto yesterday:{yesterday.strftime('%Y-%m-%d')}")
#########
fig,ax =plt.subplots(figsize=(12,6))
sns.set_style("whitegrid")
plt.figure(figsize=(12,6))
sns.lineplot(x='day',y='daily_pm_average',data=df_daily,marker='o',label='Daily PM2.5 average',ax=ax)
threshold_pm = 15
sns.scatterplot(x='day',y='daily_pm_average',data=above_threshold,color='red',s=100,label=f'PM2.5 > {threshold_pm} ug/cm3',ax=ax,zorder=5)

ax.axhline(y=threshold_pm, color='orange', linestyle='--', label=f'WHO Guideline (Daily Mean {threshold_pm} µg/m³)')

# Add labels and title
plt.title('Daily Average PM2.5 Concentration in Nairobi 2025()')
plt.xlabel('Date')
plt.ylabel('PM2.5 Concentration (µg/m³)')
plt.legend()
plt.grid(True)
plt.tight_layout() # Adjust layout to prevent labels from overlapping
st.pyplot(fig)

#plt.savefig("PM2.5 average  patterns.png")
#ozone plot
st.subheader(f"Daily average Ozone levels to:{yesterday.strftime('%Y-%m-%d')}")
fig,ax =plt.subplots(figsize=(12,6))
sns.set_style("whitegrid")
#plt.figure(figsize=(12,6))
#plot 8hr rolling mean
sns.lineplot(x='day',y='ozone_8hr_rolling_daily_mean',data=df_daily,marker='o',label='Daily Ozone 8hr rolling mean',ax=ax)
threshold = 100
above_threshold = df_daily[df_daily['ozone_8hr_rolling_daily_mean'] > threshold]
sns.scatterplot(x='day',y='ozone_8hr_rolling_daily_mean',data=above_threshold,color='red',s=100,label=f'Ozone 8hr rolling mean > {threshold} ug/cm3')
ax.axhline(y=threshold, color='orange', linestyle='--', label=f'WHO Guideline (8-hr Mean {threshold} µg/m³)')
# Add labels and title
plt.title('Daily Average Ozone Concentration in Nairobi 2025')
plt.xlabel('Date')
plt.ylabel('Ozone Concentration (µg/m³)')
ax.legend()
plt.grid(True)
plt.tight_layout() # Adjust layout to prevent labels from overlapping
st.pyplot(fig)   
