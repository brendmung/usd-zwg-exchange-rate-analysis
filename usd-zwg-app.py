import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import urllib3
import os
import PyPDF2
import re

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Initialize the HTTP manager with SSL verification disabled
http = urllib3.PoolManager(cert_reqs='CERT_NONE')

# The base URL
base_url = "https://www.rbz.co.zw/documents/Exchange_Rates/"

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
def download_pdf(url, save_path):
    try:
        response = http.request('GET', url, timeout=10)
        if response.status == 200:
            with open(save_path, 'wb') as f:
                f.write(response.data)
            return True
        else:
            return False
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

# Function to extract the MID_RATE2 for USD from a PDF
def extract_usd_mid_rate2_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = "".join(page.extract_text() for page in reader.pages)
        
        usd_pattern = re.compile(r'(USD)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)')
        match = usd_pattern.search(text)
        return float(match.group(7)) if match else None
    except Exception as e:
        print(f"Error extracting MID_RATE2 from {pdf_path}: {e}")
        return None

def update_data(csv_path):
    default_start_date = pd.Timestamp('2024-04-12')

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df['Date'] = pd.to_datetime(df['Date'])
        last_date = df['Date'].max()
    else:
        df = pd.DataFrame(columns=['Date', 'MID_RATE2', 'Filename'])
        last_date = default_start_date

    today = pd.Timestamp(datetime.now().date())
    new_data = []

    while last_date < today:
        last_date += timedelta(days=1)
        urls = generate_pdf_urls(last_date)
        
        for url in urls:
            filename = url.split('/')[-1]
            save_path = os.path.join('temp', filename)
            
            if download_pdf(url, save_path):
                mid_rate2 = extract_usd_mid_rate2_from_pdf(save_path)
                if mid_rate2 is not None:
                    new_data.append({'Date': last_date, 'MID_RATE2': mid_rate2, 'Filename': filename})
                    break  # Stop after finding the first valid PDF for this date
        
        if not new_data or new_data[-1]['Date'] != last_date:
            print(f"No data found for {last_date}")

    if new_data:
        new_df = pd.DataFrame(new_data)
        df = pd.concat([df, new_df], ignore_index=True)
        df = df.sort_values('Date').reset_index(drop=True)
        df.to_csv(csv_path, index=False)
        print(f"Data updated. {len(new_data)} new entries added.")
    else:
        print("No new data to add.")

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
@st.cache_data
def load_data():
    csv_path = 'sorted_usd_zig_rates.csv'
    df = update_data(csv_path)
    return df.drop(columns=['Filename'])

data_no_filename = load_data()

# Header
st.markdown("<div class='stHeader'>", unsafe_allow_html=True)
st.title("USD/ZWG Exchange Rate Analysis")
st.markdown("</div>", unsafe_allow_html=True)

# Sidebar for controls
st.sidebar.header("Controls")

# Date range selector
start_date = st.sidebar.date_input("Start Date", max(pd.Timestamp('2024-04-12').date(), min(data_no_filename['Date']).date()))
end_date = st.sidebar.date_input("End Date", max(data_no_filename['Date']).date())

# Filter data based on selected date range
filtered_data = data_no_filename[(data_no_filename['Date'].dt.date >= start_date) & 
                                 (data_no_filename['Date'].dt.date <= end_date)]
filtered_data = filtered_data.sort_values('Date')
filtered_data = filtered_data.drop_duplicates(subset=['Date'])


# Graph controls
line_color = st.sidebar.color_picker("Choose line color", "#1f77b4")
line_style = st.sidebar.selectbox("Choose line style", ['-', '--', '-.', ':'])
marker = st.sidebar.checkbox("Add markers")

# Main content area - Graph
st.subheader("USD/ZWG Exchange Rate Over Time")

fig, ax = plt.subplots(figsize=(20, 10))
sns.set_style("whitegrid")

if marker:
    ax.plot(filtered_data['Date'], filtered_data['MID_RATE2'], linestyle=line_style, color=line_color, 
            marker='o', markersize=6, markerfacecolor='orange', markeredgewidth=2, markeredgecolor='black')
else:
    ax.plot(filtered_data['Date'], filtered_data['MID_RATE2'], linestyle=line_style, color=line_color, linewidth=2)

ax.set_title('USD/ZWG Exchange Rate Over Time', fontsize=20, fontweight='bold')
ax.set_xlabel('Date', fontsize=16, fontweight='bold')
ax.set_ylabel('USD per ZWG', fontsize=16, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)

ax.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

st.pyplot(fig)

# Tables and Statistics
col1, col2 = st.columns(2)

# Left column: Data table
with col1:
    st.subheader("USD/ZWG Exchange Rate Data")
    st.info("Note: As the rate increases, more ZWG is required to buy 1 USD, indicating ZWG depreciation.")
    
    table_data = filtered_data.sort_values(by='Date', ascending=False)
    st.dataframe(table_data.style.highlight_max(axis=0), height=400)

# Right column: Summary statistics
with col2:
    st.subheader("Summary Statistics")
    summary = filtered_data['MID_RATE2'].describe()
    st.table(summary.round(4))
    
    if len(filtered_data) > 1:
        total_change = filtered_data['MID_RATE2'].iloc[-1] - filtered_data['MID_RATE2'].iloc[0]
        total_days = (filtered_data['Date'].iloc[-1] - filtered_data['Date'].iloc[0]).days
        daily_change = total_change / total_days
        
        st.metric("Average daily change", f"{daily_change:.4f} USD/ZWG")
        st.metric("Total change over period", f"{total_change:.4f} USD/ZWG")

# Download and update buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("Download Filtered Data as CSV"):
        csv = table_data.to_csv(index=False)
        st.download_button(
            label="Click here to download",
            data=csv,
            file_name="filtered_usd_zig_rates.csv",
            mime="text/csv",
        )

with col2:
    if st.button("Update Data"):
        with st.spinner("Updating data... This may take a moment."):
            updated_data = update_data('sorted_usd_zig_rates.csv')
        st.success("Data updated successfully!")
        st.experimental_rerun()

st.sidebar.info(f"Data is automatically updated daily. Last update: {data_no_filename['Date'].max().date()}")
