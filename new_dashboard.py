import os
import re
import tempfile
import requests
import streamlit as st
from bs4 import BeautifulSoup
import whisper
from transformers import pipeline
import csv

# Ensure temp folder exists early
os.makedirs("temp_recordings", exist_ok=True)

# Load summarization pipeline once; handle model load errors
try:
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
except Exception as e:
    st.error(f"Error loading summarization model: {e}")
    st.stop()

def sanitize_filename(s):
    return re.sub(r'[<>:"/\\|?* ]', '_', s)

def parse_html_calls(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    calls = []

    for row in soup.select("tr.recording"):
        data_id = row.get("data-id")
        date_td = row.find('td', class_='date')
        full_date = date_td.text.strip() if date_td else ""
        rec_td = row.find('td', class_='rec')
        rec_number = rec_td.find('span', class_='phonenumber').text.strip() if rec_td else ""
        from_td = row.find('td', class_='from')
        from_number = from_td.find('span', class_='phonenumber').text.strip().replace(" ", "") if from_td else ""
        to_td = row.find_all('td')[4] if len(row.find_all('td')) > 4 else None
        to_number = to_td.find('span', class_='phonenumber').text.strip().replace(" ", "") if to_td else ""

        if rec_number == "*200":
            user_tag = "Vikki"
        elif rec_number == "*201":
            user_tag = "Assistant"
        else:
            user_tag = "UnknownUser"

        if data_id:
            calls.append({
                "data_id": data_id,
                "date_time": full_date,
                "from_number": from_number,
                "to_number": to_number,
                "user_tag": user_tag
            })

    return calls

def download_audio(session, call):
    fn = f"{sanitize_filename(call['date_time'])}_{call['from_number']}_{call['to_number']}_{call['user_tag']}_{call['data_id']}.mp3"
    filepath = os.path.join("temp_recordings", fn)
    if os.path.exists(filepath):
        return filepath
    url = f"https://controlpanel.voipfone.co.uk/api/srv?callRecordingsGetFile/{call['data_id']}.mp3"
    resp = session.get(url, stream=True)
    resp.raise_for_status()
    with open(filepath, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    return filepath

def transcribe_audio(model, path):
    try:
        result = model.transcribe(path, task="translate")
        text = result.get("text", "")
        return text.strip()
    except Exception as e:
        st.error(f"Transcription error on {path}: {e}")
        return ""

def summarize_text(text):
    if not text:
        return "No transcript available"
    # Limit size to ~1000 chars to avoid errors in summarizer
    text = text[:1000]
    try:
        summaries = summarizer(text, max_length=80, min_length=30, do_sample=False)
        return summaries[0]['summary_text']
    except Exception as e:
        st.error(f"Summarization error: {e}")
        return "Error in summarization"

st.title("Mortgage Call Downloader, Transcriber & Summarizer")

uploaded_html = st.file_uploader("Upload calls HTML file (Voipfone page)", type=["html"])
cookie_name = st.text_input("Session Cookie name (e.g. voipfone_auth)")
cookie_value = st.text_input("Session Cookie value", type="password")

if uploaded_html and cookie_name and cookie_value:
    html_string = uploaded_html.read().decode('utf-8')
    calls = parse_html_calls(html_string)
    st.write(f"Detected {len(calls)} calls in HTML.")

    if st.button("Download recordings & process"):
        session = requests.Session()
        session.cookies.set(cookie_name, cookie_value, domain=".voipfone.co.uk")

        try:
            model = whisper.load_model("tiny")
        except Exception as e:
            st.error(f"Error loading Whisper model: {e}")
            st.stop()

        csv_path = "calls_summary.csv"
        # Open CSV for append, write header if missing
        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["Date (+time)", "From", "To", "Summary"])
                writer.writeheader()

        progress_bar = st.progress(0)

        for i, call in enumerate(calls):
            st.write(f"Processing call {i+1}/{len(calls)}: {call['data_id']}")
            try:
                mp3_path = download_audio(session, call)
                transcript = transcribe_audio(model, mp3_path)
                summary = summarize_text(transcript)

                with open(csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=["Date (+time)", "From", "To", "Summary"])
                    writer.writerow({
                        "Date (+time)": call["date_time"],
                        "From": call["from_number"],
                        "To": call["to_number"],
                        "Summary": summary
                    })
            except Exception as e:
                st.error(f"Error processing call {call['data_id']}: {e}")
            progress_bar.progress((i + 1) / len(calls))

        st.success("Processing complete! Download CSV below.")
        with open(csv_path, "r", encoding="utf-8") as f:
            st.download_button("Download CSV", data=f.read(), file_name=csv_path)
