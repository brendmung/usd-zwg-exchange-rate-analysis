import pandas as pd
import requests
import os
import holidays
import PyPDF2
from datetime import datetime, timedelta, date
import re
import json
import sys
import time

# Disable SSL warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Create a persistent session with browser-like headers
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'none',
    'Sec-Ch-Ua': '"Google Chrome";v="119", "Chromium";v="119", "Not?A_Brand";v="24"',
    'Sec-Ch-Ua-Mobile': '?0',
    'Sec-Ch-Ua-Platform': '"Windows"',
})

# The base URL
base_url = "https://www.rbz.co.zw/documents/Exchange_Rates/"

# Ensure the 'temp' directory exists
os.makedirs('temp', exist_ok=True)

def is_market_closed_today():
    today = date.today()
    zw_holidays = holidays.ZW()  # Zimbabwean holidays
    return today.weekday() >= 5 or today in zw_holidays  # 5 and 6 are Saturday and Sunday

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

def download_pdf(url, date):
    try:
        print(f"Attempting to download: {url}")
        
        # Add a small delay to be more respectful to the server
        time.sleep(1)
        
        response = session.get(url, timeout=30, verify=False)
        
        if response.status_code == 200:
            # Check if it's actually a PDF and not HTML (bot detection page)
            if not response.content.startswith(b'%PDF'):
                if b'captcha' in response.content.lower() or b'radware' in response.content.lower():
                    print(f"Bot detection triggered for {url}")
                    return False
                print(f"Downloaded file is not a valid PDF: {url}")
                return False
            
            day = date.day
            month_name = date.strftime("%B")
            year = date.year
            correct_filename = f"RATES_{day:02d}_{month_name.upper()}_{year}.pdf"
            save_path = os.path.join('temp', correct_filename)

            with open(save_path, 'wb') as f:
                f.write(response.content)
            print(f"Successfully downloaded: {correct_filename} ({len(response.content)} bytes)")
            return True
        else:
            print(f"Failed to download {url}, status: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Request error downloading {url}: {str(e)}")
        return False
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        return False

def extract_usd_rates_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = "".join(page.extract_text() for page in reader.pages)
        
        print(f"Extracted text length: {len(text)}")
        
        usd_pattern = re.compile(r'(USD)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)')
        match = usd_pattern.search(text)
        if match:
            rates = {
                'BID': float(match.group(5)),
                'ASK': float(match.group(6)),
                'MID_RATE': float(match.group(7))
            }
            print(f"Extracted rates: {rates}")
            return rates
        else:
            print("No USD rates found in PDF")
            return None
    except Exception as e:
        print(f"Error extracting rates from {pdf_path}: {str(e)}")
        return None

def get_last_update_info():
    """Get information about the last update from a metadata file"""
    metadata_file = 'update_metadata.json'
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                return metadata
        except Exception as e:
            print(f"Error reading metadata: {str(e)}")
            return None
    return None

def save_last_update_info(last_date, update_time):
    """Save information about the last update"""
    metadata = {
        'last_date': last_date.isoformat() if hasattr(last_date, 'isoformat') else str(last_date),
        'last_update_time': update_time.isoformat(),
        'update_count': 1
    }
    
    # Try to read existing metadata to increment update count
    existing_metadata = get_last_update_info()
    if existing_metadata:
        metadata['update_count'] = existing_metadata.get('update_count', 0) + 1
    
    try:
        with open('update_metadata.json', 'w') as f:
            json.dump(metadata, f)
        print(f"Saved metadata: {metadata}")
    except Exception as e:
        print(f"Error saving metadata: {str(e)}")

def update_data():
    """Update exchange rate data"""
    csv_path = 'sorted_usd_zig_rates.csv'
    default_start_date = pd.Timestamp('2025-09-30').date()

    print("Starting data update...")
    
    # Check if market is closed
    if is_market_closed_today():
        print("Market is closed today, skipping update")
        return

    # Load existing data
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        print("Loading existing data...")
        df = pd.read_csv(csv_path)
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        if 'BID' not in df.columns:
            df['BID'] = None
        if 'ASK' not in df.columns:
            df['ASK'] = None
        last_date = df['Date'].max()
        print(f"Last date in existing data: {last_date}")
    else:
        print("Creating new dataframe...")
        df = pd.DataFrame(columns=['Date', 'BID', 'ASK', 'MID_RATE', 'Filename'])
        last_date = default_start_date

    today = datetime.now().date()
    new_data = []
    
    print(f"Checking for new data from {last_date} to {today}")
    
    # Generate dates to check
    dates_to_check = []
    check_date = last_date
    while check_date < today:
        check_date += timedelta(days=1)
        # Skip weekends and holidays
        if check_date.weekday() < 5 and check_date not in holidays.ZW():
            dates_to_check.append(check_date)
    
    print(f"Will check {len(dates_to_check)} dates: {dates_to_check}")
    
    for check_date in dates_to_check:
        print(f"\nChecking data for {check_date}")
        urls = generate_pdf_urls(check_date)
        
        found_data = False
        for url in urls:
            if download_pdf(url, check_date):
                correct_filename = f"RATES_{check_date.day:02d}_{check_date.strftime('%B').upper()}_{check_date.year}.pdf"
                save_path = os.path.join('temp', correct_filename)
                
                rates = extract_usd_rates_from_pdf(save_path)
                if rates is not None:
                    new_data.append({
                        'Date': check_date,
                        'BID': rates['BID'],
                        'ASK': rates['ASK'],
                        'MID_RATE': rates['MID_RATE'],
                        'Filename': correct_filename
                    })
                    print(f"Added data for {check_date}: {rates}")
                    found_data = True
                    break  # Stop after finding the first valid PDF for this date
                
                # Clean up the downloaded file
                try:
                    os.remove(save_path)
                except:
                    pass
        
        if not found_data:
            print(f"No data found for {check_date}")
        
        # Add a delay between date checks to avoid overwhelming the server
        time.sleep(2)

    if new_data:
        print(f"\nFound {len(new_data)} new entries")
        new_df = pd.DataFrame(new_data)
        df = pd.concat([df, new_df], ignore_index=True)
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Remove filename column for CSV output
        df_to_save = df.drop(columns=['Filename']) if 'Filename' in df.columns else df
        df_to_save.to_csv(csv_path, index=False)
        
        # Save update metadata
        save_last_update_info(df['Date'].max(), datetime.now())
        
        print(f"Data updated successfully! {len(new_data)} new entries added.")
        print(f"Total entries in dataset: {len(df)}")
        
        # Display the new data
        for entry in new_data:
            print(f"  {entry['Date']}: {entry['MID_RATE']}")
            
    else:
        print("No new data found")
        # Still save metadata even if no new data
        save_last_update_info(last_date, datetime.now())

    # Clean up temp directory
    try:
        for file in os.listdir('temp'):
            os.remove(os.path.join('temp', file))
        os.rmdir('temp')
    except:
        pass

if __name__ == "__main__":
    try:
        update_data()
        print("Script completed successfully")
    except Exception as e:
        print(f"Script failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
