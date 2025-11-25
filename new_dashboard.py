import os
import re
import tempfile
import requests
import streamlit as st
from bs4 import BeautifulSoup
import whisper
import openai
import csv
from tqdm import tqdm

# Setup OpenAI key from secrets or env variables
openai.api_key = st.secrets["OPENAI_API_KEY"]

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
    if not os.path.exists("temp_recordings"):
        os.makedirs("temp_recordings")
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
    result = model.transcribe(path, task="translate")
    return result["text"]

def summarize_text(text):
    prompt = (
        "You are an expert mortgage advisor assistant. "
        "Summarize this client call transcript briefly (2-5 sentences for short calls, 5-10 for longer):\n\n"
        f"{text}\n\nSummary:"
    )
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant specialized in mortgage advising."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()

st.title("Mortgage Call Downloader, Transcriber & Summarizer")

uploaded_html = st.file_uploader("Upload calls HTML file (Voipfone page)", type=["html"])
cookie_name = st.text_input("Session Cookie name (e.g. voipfone_auth)")
cookie_value = st.text_input("Session Cookie value", type="password")

if uploaded_html and cookie_name and cookie_value:
    html_string = uploaded_html.read().decode('utf-8')
    calls = parse_html_calls(html_string)
    st.write(f"Detected {len(calls)} calls in HTML.")

    if st.button("Download recordigs & process"):
        session = requests.Session()
        session.cookies.set(cookie_name, cookie_value, domain=".voipfone.co.uk")

        model = whisper.load_model("tiny")
        results = []

        progress_bar = st.progress(0)
        for i, call in enumerate(calls):
            st.write(f"Processing call {i+1}/{len(calls)}: {call['data_id']}")
            try:
                mp3_path = download_audio(session, call)
                transcript = transcribe_audio(model, mp3_path)
                summary = summarize_text(transcript)
                results.append({
                    "Date (+time)": call["date_time"],
                    "From": call["from_number"],
                    "To": call["to_number"],
                    "Summary": summary
                })
            except Exception as e:
                st.error(f"Error processing call {call['data_id']}: {e}")

            progress_bar.progress((i + 1) / len(calls))

        if results:
            csv_path = "calls_summary.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["Date (+time)", "From", "To", "Summary"])
                writer.writeheader()
                writer.writerows(results)
            st.success(f"Processing complete! Download CSV below.")
            st.download_button("Download CSV", data=open(csv_path, "r", encoding="utf-8").read(), file_name=csv_path)



