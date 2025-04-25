import streamlit as st
st.set_page_config(page_title="📉 Head & Shoulders Screener", layout="wide")

import pandas as pd
import yfinance as yf
import datetime
import os
import gspread
from google.oauth2.service_account import Credentials
from pattern_detector import detect_head_and_shoulders

# Google Sheets setup
SHEET_NAME = "HnS_Pattern_Log"
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

def get_gsheet_client():
    creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=SCOPES)
    client = gspread.authorize(creds)
    return client

def append_to_gsheet(matches):
    try:
        client = get_gsheet_client()
        sheet = client.open(SHEET_NAME).sheet1
        for row in matches.values.tolist():
            sheet.append_row([datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")] + row)
        st.success("✅ Appended results to Google Sheets")
    except Exception as e:
        st.error(f"❌ Failed to append to Google Sheets: {e}")

if os.path.exists("nifty100.txt"):
    with open("nifty100.txt") as f:
        nifty100_symbols = [line.strip() for line in f.readlines() if line.strip()]
else:
    st.error("❌ 'nifty100.txt' file not found.")
    st.stop()

st.title("📉 Head & Shoulders & Inverse H&S Screener")
st.caption("Scans NIFTY100 stocks for potential Head and Shoulders or Inverse H&S patterns using 90 daily candles")

confidence_threshold = st.slider("Minimum Pattern Match Confidence (%)", min_value=50, max_value=100, value=70)
pattern_type = st.selectbox("Pattern Type", ["Head & Shoulders", "Inverse H&S"])

progress = st.progress(0)
results = []
skipped = []

for i, symbol in enumerate(nifty100_symbols):
    progress.progress((i + 1) / len(nifty100_symbols))
    try:
        df = yf.download(symbol, period="120d", interval="1d").dropna()
        if df.empty or df['Close'].isnull().sum() > 5:
            skipped.append((symbol, "Empty or NaNs"))
            continue
        df = df.tail(90)
        if len(df) < 90:
            skipped.append((symbol, "Insufficient data"))
            continue

        inverse = True if pattern_type == "Inverse H&S" else False
        match, score, _ = detect_head_and_shoulders(df, inverse=inverse)

        if bool(match) and float(score) >= confidence_threshold:
            results.append({
                "Symbol": symbol.replace(".NS", ""),
                "Confidence %": round(score, 2),
                "Pattern": pattern_type
            })
        else:
            skipped.append((symbol, "Pattern not matched"))
    except Exception as e:
        skipped.append((symbol, str(e)))

if results:
    st.success(f"✅ Found {len(results)} matching stocks")
    matches_df = pd.DataFrame(results).sort_values(by="Confidence %", ascending=False).reset_index(drop=True)
    st.dataframe(matches_df, use_container_width=True)

    if st.button("📤 Append to Google Sheet Log"):
        append_to_gsheet(matches_df)

    csv = matches_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name="matched_patterns.csv",
        mime='text/csv')
else:
    st.warning("😕 No stocks matched the selected pattern with current threshold.")

if skipped:
    with st.expander("🧾 View Skipped Stocks"):
        st.dataframe(pd.DataFrame(skipped, columns=["Symbol", "Reason"]))
