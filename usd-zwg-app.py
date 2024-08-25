import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import urllib3
import os
import holidays
import PyPDF2
import mplcursors
from datetime import date
import re
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from multiprocessing import Process

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

csv_path = 'sorted_usd_zig_rates.csv'

@app.get("/api/exchange_rates")
def read_exchange_rates():
    data = load_data()
    return data.to_dict(orient='records')

@app.get("/api/market_status")
def get_market_status():
    return {"is_closed": is_market_closed_today()}

@app.get("/api/latest_rate")
def get_latest_rate():
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        latest_date = df['Date'].max()
        latest_rate = df.loc[df['Date'] == latest_date, 'MID_RATE'].iloc[0]
        return {
            "latest_date": latest_date.strftime('%Y-%m-%d'),
            "latest_rate": round(latest_rate, 4)
        }
    else:
        return {"error": "Data file not found"}

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Initialize the HTTP manager with SSL verification disabled
http = urllib3.PoolManager(cert_reqs='CERT_NONE')

# The base URL
base_url = "https://www.rbz.co.zw/documents/Exchange_Rates/"

# Ensure the 'temp' directory exists
os.makedirs('temp', exist_ok=True)

def is_market_closed_today():
    today = date.today()
    zw_holidays = holidays.ZW()  # Zimbabwean holidays
    return today.weekday() >= 5 or today in zw_holidays  # 5 and 6 are Saturday and Sunday


# Function to generate possible PDF URLs
def generate_pdf_urls(date):
    day = date.day
    month_name = date.strftime("%B")
    year = date.year
    
    day_formats = [f"{day:02d}"] if day >= 10 else [f"{day}", f"{day:02d}"]
    
    urls = []
    for day_format in day_formats:
        url1 = f"{base_url}{year}/{month_name}/RATES_{day_format}_{month_name.upper()}_{year}.pdf"
        url2 = f"{base_url}{year}/{month_name}/RATES_{day_format}_{month_name.upper()}_{year}-INTERBANK_RATE.pdf"
        urls.extend([url1, url2])
    return urls

# Function to check if a URL exists and download it
def download_pdf(url, date):
    try:
        response = http.request('GET', url, timeout=10)
        if response.status == 200:
            day = date.day
            month_name = date.strftime("%B")
            year = date.year
            correct_filename = f"RATES_{day:02d}_{month_name.upper()}_{year}.pdf"
            save_path = os.path.join('temp', correct_filename)

            with open(save_path, 'wb') as f:
                f.write(response.data)
            return True
        else:
            return False
    except Exception as e:
        #st.warning(f"Error downloading {url}: {e}")
        st.warning("Fetching error occured!")
        return False

# Function to extract the MID_RATE2 for USD from a PDF
def extract_usd_rates_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = "".join(page.extract_text() for page in reader.pages)
        
        usd_pattern = re.compile(r'(USD)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)')
        match = usd_pattern.search(text)
        if match:
            return {
                'BID': float(match.group(5)),
                'ASK': float(match.group(6)),
                'MID_RATE': float(match.group(7))
            }
        return None
    except Exception as e:
        #st.warning(f"Error extracting rates from {pdf_path}: {e}")
        st.warning("Data extraction error occured!")
        return None

def update_data(csv_path):
    default_start_date = pd.Timestamp('2024-04-11').date()

    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        df = pd.read_csv(csv_path)
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        if 'BID' not in df.columns:
            df['BID'] = None
        if 'ASK' not in df.columns:
            df['ASK'] = None
        last_date = df['Date'].max()
    else:
        df = pd.DataFrame(columns=['Date', 'BID', 'ASK', 'MID_RATE', 'Filename'])
        last_date = default_start_date

    today = datetime.now().date()

    new_data = []

    while last_date < today:
        last_date += timedelta(days=1)
        urls = generate_pdf_urls(last_date)
        
        for url in urls:
            if download_pdf(url, last_date):
                correct_filename = f"RATES_{last_date.day:02d}_{last_date.strftime('%B').upper()}_{last_date.year}.pdf"
                save_path = os.path.join('temp', correct_filename)
                
                rates = extract_usd_rates_from_pdf(save_path)
                if rates is not None:
                    new_data.append({
                        'Date': last_date,
                        'BID': rates['BID'],
                        'ASK': rates['ASK'],
                        'MID_RATE': rates['MID_RATE'],
                        'Filename': correct_filename
                    })
                    break  # Stop after finding the first valid PDF for this date
        
        #if not new_data or new_data[-1]['Date'] != last_date:
        #    st.info(f"No data found for {last_date}")

    if new_data:
        new_df = pd.DataFrame(new_data)
        df = pd.concat([df, new_df], ignore_index=True)
        df = df.sort_values('Date').reset_index(drop=True)
        df.to_csv(csv_path, index=False)
        #st.success(f"Data updated. {len(new_data)} new entries added.")
    #else:
        #st.info("No new data to add.")

    return df

st.set_page_config(page_title="USD/ZWG Exchange Rate Analysis", layout="wide")

# Custom CSS to improve the interface
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stButton>button {
        width: 100%;
    }
    .stDateInput>div>div>input {
        min-height: 40px;
    }
    .stHeader {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data():
    csv_path = 'sorted_usd_zig_rates.csv'
    df = update_data(csv_path)
    # No need to convert 'Date' column here, it's already in date format
    return df.drop(columns=['Filename'])

data_no_filename = load_data()

# Header
st.markdown("<div class='stHeader'>", unsafe_allow_html=True)
st.title("USD/ZWG Exchange Rate Analysis")
st.markdown("</div>", unsafe_allow_html=True)

# Sidebar for controls
st.sidebar.header("Controls")

# Determine the minimum and maximum dates for the date range picker
if not data_no_filename.empty:
    min_date = data_no_filename['Date'].min()
    max_date = data_no_filename['Date'].max()
else:
    min_date = pd.Timestamp('2024-04-11').date()
    max_date = datetime.now().date()

one_month_ago = max(min_date, max_date - timedelta(days=30))

# Set the default date range for the graph (1 month ago to today)
start_date = st.sidebar.date_input("Start Date", value=one_month_ago, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("End Date", value=max_date, min_value=start_date, max_value=max_date)

# Filter data based on selected date range
filtered_data = data_no_filename[(data_no_filename['Date'] >= start_date) & 
                                 (data_no_filename['Date'] <= end_date)]
filtered_data = filtered_data.sort_values('Date')
filtered_data = filtered_data.drop_duplicates(subset=['Date'])

# Graph controls
line_color = st.sidebar.color_picker("Choose line color", "#1f77b4")
line_style = st.sidebar.selectbox("Choose line style", ['-', '--', '-.', ':'])
marker = st.sidebar.checkbox("Add markers")

# Main content area - Graph

if is_market_closed_today():
    st.markdown("<h3 style='color: red;'>Market Closed Today</h3>", unsafe_allow_html=True)
else:
    st.subheader("Current Exchange Rate:")

# Display current rate
if not filtered_data.empty:
    latest_date = filtered_data['Date'].max()
    latest_rate = filtered_data.loc[filtered_data['Date'] == latest_date, 'MID_RATE'].iloc[0]
    st.markdown(f"<h1 style='color: #1f77b4;'>1 USD = {latest_rate:.4f} ZWG</h1>", unsafe_allow_html=True)
    st.markdown(f"<p>Last updated: {latest_date}</p>", unsafe_allow_html=True)
   
# Add some space
st.markdown("<br>", unsafe_allow_html=True)

# Main content area - Graph
fig, ax = plt.subplots(figsize=(20, 10))
sns.set_style("whitegrid")
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Plotting the line
ax.plot(filtered_data['Date'], filtered_data['MID_RATE'], linestyle=line_style, color=line_color, linewidth=2,
        marker='o' if marker else None, markersize=6, markerfacecolor='orange', markeredgewidth=2, markeredgecolor='black',
        label='ZWG')

# Apply shading below the line
ax.fill_between(filtered_data['Date'], filtered_data['MID_RATE'], color=line_color, alpha=0.2)

# Determine the range for Y-axis limits
y_min = filtered_data['MID_RATE'].min()
y_max = filtered_data['MID_RATE'].max()
buffer = (y_max - y_min) * 0.05
ax.set_ylim(y_min - buffer, y_max + buffer)

# Set x-axis limit to start from the first date in the filtered data
ax.set_xlim(left=filtered_data['Date'].min())

# Remove extra space between axis and plot
ax.margins(x=0)

ax.set_title('USD/ZWG Exchange Rate', fontsize=20, fontweight='bold')
ax.set_xlabel('Date', fontsize=16, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)

ax.grid(True, linestyle='--', alpha=0.6)
ax.legend()

# Enable hover effect
cursor = mplcursors.cursor(ax, hover=True)
@cursor.connect("add")
def on_add(sel):
    x, y = sel.target
    date = mdates.num2date(x).strftime('%Y-%m-%d')
    sel.annotation.set_text(f"Date: {date}\nRate: {y:.4f}")
    sel.annotation.get_bbox_patch().set(fc="white", alpha=0.8)

st.pyplot(fig)
plt.close(fig)


# Tables and Statistics
col1, col2 = st.columns(2)

# Left column: Data table
with col1:
    st.subheader("USD/ZWG Exchange Rate Data")
    st.info("Note: As the rates increase, more ZWG is required to buy 1 USD, indicating ZWG depreciation.")
    
    table_data = filtered_data.sort_values(by='Date', ascending=False)
    formatted_table_data = table_data.copy()
    for column in ['BID', 'ASK', 'MID_RATE']:
        formatted_table_data[column] = formatted_table_data[column].apply(lambda x: f"{x:.4f}".rstrip('.'))
    
    # Reset the index to start from 1 and make it visible
    formatted_table_data = formatted_table_data.reset_index(drop=True)
    formatted_table_data.index = formatted_table_data.index + 1
    
    st.dataframe(formatted_table_data.style.highlight_max(axis=0), height=400)

# Right column: Summary statistics
with col2:
    st.subheader("Summary Statistics")
    
    rates_to_show = ['MID_RATE', 'BID', 'ASK']
    stats_to_show = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    
    stats_data = {}
    for rate in rates_to_show:
        if rate in filtered_data.columns and not filtered_data[rate].isnull().all():
            stats = filtered_data[rate].describe()
            stats_data[rate] = stats[stats_to_show]
    
    if stats_data:
        combined_stats = pd.DataFrame(stats_data)
        combined_stats.index.name = 'Statistic'
        combined_stats = combined_stats.round(4)
        
        st.table(combined_stats)
    else:
        st.info("No statistical data available for the selected date range.")
    
    if len(filtered_data) > 1:
        st.subheader("Rate Changes")
        changes_data = {}
        for rate in rates_to_show:
            if rate in filtered_data.columns and not filtered_data[rate].isnull().all():
                total_change = filtered_data[rate].iloc[-1] - filtered_data[rate].iloc[0]
                total_days = (filtered_data['Date'].iloc[-1] - filtered_data['Date'].iloc[0]).days
                daily_change = total_change / max(total_days, 1)  # Avoid division by zero
                
                changes_data[rate] = {
                    'Average Daily Change': f"{daily_change:.4f}",
                    'Total Change': f"{total_change:.4f}"
                }
        
        if changes_data:
            changes_df = pd.DataFrame(changes_data).T
            changes_df.index.name = 'Rate'
            st.table(changes_df)
        else:
            st.info("No change data available for the selected date range.")
            
# Download button
if st.button("Download Filtered Data as CSV"):
    csv = filtered_data.to_csv(index=False)
    st.download_button(
        label="Click here to download",
        data=csv,
        file_name="filtered_usd_zwg_rates.csv",
        mime="text/csv",
    )

st.sidebar.info(f"Data is automatically updated daily. Last update: {data_no_filename['Date'].max().strftime('%Y-%m-%d')}")


def run_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    # Start FastAPI in a separate process
    Process(target=run_fastapi).start()
    
    # Run Streamlit
    import streamlit.web.bootstrap as bootstrap
    bootstrap.run(file=__file__, command_line=[])
