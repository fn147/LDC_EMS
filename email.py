import csv
import email
import imaplib
import io
import os
import time
from email.header import decode_header
from pathlib import Path

import httpx
import pdfplumber
import streamlit as st
from openai import OpenAI

MODEL_NAME = "gpt-4.1"
MAX_RETRIES = 5
RETRY_SLEEP = 3
DEFAULT_PROMPT = """You are reading a microbiological laboratory test report.

Your tasks:
1. Find the FINAL TEST RESULT requested by the user.
2. Find the SAMPLE DATE (date when the sample was taken).
3. Find the SAMPLE TIME if available.

Decision rules (VERY IMPORTANT):
- Output "+" ONLY if the report explicitly states the target is detected
  (e.g. "nachgewiesen", "detected", "positive").
- Output "-" ONLY if the report explicitly states the target is NOT detected
  (e.g. "nicht nachgewiesen", "n.n.", "not detected", "negative").
- Ignore symbols (+ / -) unless they are clearly part of the final test result.
- Ignore method descriptions, legends, footnotes, and example text.

Date & Time rules:
- Extract the sampling date (e.g. "Probenahme", "Sampling Date", "Sample Date").
- Extract the sampling time if explicitly stated.
- Return date in format YYYY-MM-DD if possible.
- Return time in format HH:MM (24h) if possible.
- If no date is found, return "unknown".
- If no time is found, return "unknown".

Output format (STRICT JSON ONLY):
{
  "result": "+ or -",
  "reason": "very short explanation",
  "evidence": "exact line from report",
  "sample_date": "YYYY-MM-DD or unknown",
  "sample_time": "HH:MM or unknown"
}

Rules:
- Be very brief
- Do not explain laboratory methods
- Do not interpret Ct values
- Do not add extra text
- write everything in English
"""

http_client = httpx.Client(timeout=60, verify=False)


def get_openai_api_key() -> str:
    # Streamlit Cloud Secrets:
    # OPENAI_API_KEY = "sk-..."
    return st.secrets.get("OPENAI_API_KEY", "").strip()


def decode_mime_header(value):
    if not value:
        return ""
    parts = decode_header(value)
    decoded = []
    for text, encoding in parts:
        if isinstance(text, bytes):
            decoded.append(text.decode(encoding or "utf-8", errors="replace"))
        else:
            decoded.append(text)
    return "".join(decoded)


def get_text_body(msg):
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = part.get("Content-Disposition", "")
            if content_type == "text/plain" and "attachment" not in content_disposition:
                payload = part.get_payload(decode=True) or b""
                charset = part.get_content_charset() or "utf-8"
                return payload.decode(charset, errors="replace")
        return ""
    payload = msg.get_payload(decode=True) or b""
    charset = msg.get_content_charset() or "utf-8"
    return payload.decode(charset, errors="replace")


def save_attachments(msg, base_dir, msg_id):
    saved_files = []
    if not msg.is_multipart():
        return saved_files

    safe_dir = Path(base_dir) / msg_id
    safe_dir.mkdir(parents=True, exist_ok=True)

    for part in msg.walk():
        filename = part.get_filename()
        if not filename:
            continue
        filename = decode_mime_header(filename)
        payload = part.get_payload(decode=True)
        if not payload:
            continue
        path = safe_dir / filename
        if path.suffix.lower() == ".pdf" and path.exists():
            continue
        path.write_bytes(payload)
        saved_files.append(str(path))
    return saved_files


def extract_text_from_pdf(data: bytes) -> str:
    pages = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n".join(pages)


def extract_text_from_path(path: Path) -> str:
    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n".join(pages)


def build_prompt(prompt_template: str, text: str) -> str:
    if "{text}" in prompt_template:
        try:
            return prompt_template.format(text=text)
        except Exception:
            return f"{prompt_template}\n\nReport text:\n{text}"
    return f"{prompt_template}\n\nReport text:\n{text}"


def llm_analyze(text: str, prompt_template: str, client: OpenAI) -> str:
    prompt = build_prompt(prompt_template, text)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.responses.create(
                model=MODEL_NAME,
                input=prompt,
                max_output_tokens=200,
            )
            return response.output_text.strip()
        except Exception:
            if attempt == MAX_RETRIES:
                raise
            time.sleep(RETRY_SLEEP)


def fetch_messages(
    gmail_user,
    gmail_app_password,
    sender_filter,
    attachments_dir,
    save_attachments_enabled,
):
    mail = imaplib.IMAP4_SSL("imap.gmail.com", timeout=30)
    try:
        mail.login(gmail_user, gmail_app_password)
        mail.select("inbox")
        status, data = mail.search(None, f'(FROM "{sender_filter}")')
        if status != "OK":
            raise RuntimeError("Failed to search mailbox.")

        email_ids = data[0].split() if data and data[0] else []
        selected_ids = email_ids[::-1]

        results = []
        for msg_id in selected_ids:
            status, msg_data = mail.fetch(msg_id, "(RFC822)")
            if status != "OK":
                continue
            raw_msg = msg_data[0][1]
            msg = email.message_from_bytes(raw_msg)
            subject = decode_mime_header(msg.get("Subject"))
            from_ = decode_mime_header(msg.get("From", "(unknown sender)"))
            date_ = msg.get("Date", "(no date)")
            body = get_text_body(msg)

            saved = []
            if save_attachments_enabled:
                saved = save_attachments(msg, attachments_dir, msg_id.decode(errors="replace"))

            results.append(
                {
                    "id": msg_id.decode(errors="replace"),
                    "from": from_,
                    "subject": subject,
                    "date": date_,
                    "body": body,
                    "attachments": saved,
                }
            )

        return results, len(email_ids)
    finally:
        try:
            mail.logout()
        except Exception:
            pass


def main():
    st.set_page_config(page_title="Email Attachment Fetcher", layout="wide")
    st.title("Email Attachment Fetcher")

    attachments_root = Path(__file__).resolve().parent / "attachments"

    st.subheader("PDF Source")
    source_mode = st.radio(
        "Choose how to provide PDFs",
        options=["Upload PDFs", "Fetch from email"],
        horizontal=True,
    )

    gmail_user = ""
    sender_filter = ""
    save_attachments_enabled = True
    password_input = ""
    fetch_clicked = False
    uploads = []

    if source_mode == "Fetch from email":
        with st.sidebar:
            st.header("Connection")
            gmail_user = st.text_input("Gmail address", value="f.norouzi147@gmail.com")
            sender_filter = st.text_input("Sender filter", value="fatemeh.norouzi@ldc.com")
            save_attachments_enabled = st.checkbox("Save attachments", value=True)

            st.header("Credentials")
            password_input = st.text_input("Gmail app password", type="password")
            st.caption("If empty, the app uses `GMAIL_APP_PASSWORD` from the environment.")

            fetch_clicked = st.button("Fetch emails", type="primary")
    else:
        uploads = st.file_uploader(
            "Drop PDF files here",
            type=["pdf"],
            accept_multiple_files=True,
        )
        if uploads:
            st.success(f"Received {len(uploads)} PDF file(s).")
        else:
            st.info("Drop one or more PDF files to continue.")

    if source_mode == "Fetch from email" and fetch_clicked:
        gmail_app_password = password_input or os.getenv("GMAIL_APP_PASSWORD")
        if not gmail_user or not sender_filter:
            st.error("Please provide Gmail address and sender filter.")
            return
        if not gmail_app_password:
            st.error("Missing Gmail app password.")
            return

        with st.spinner("Fetching messages..."):
            try:
                results, total_found = fetch_messages(
                    gmail_user=gmail_user,
                    gmail_app_password=gmail_app_password,
                    sender_filter=sender_filter,
                    attachments_dir=str(attachments_root),
                    save_attachments_enabled=save_attachments_enabled,
                )
            except Exception as exc:
                st.error(f"Fetch failed: {exc}")
                return

        st.success(f"Found {total_found} message(s). Showing {len(results)}.")
        if not results:
            st.info("No messages matched the filter.")
        else:
            st.subheader("Messages")
            st.dataframe(
                [
                    {
                        "From": item["from"],
                        "Subject": item["subject"],
                        "Date": item["date"],
                        "Attachments": len(item["attachments"]),
                    }
                    for item in results
                ],
                use_container_width=True,
            )

            for item in results:
                subject = item["subject"] or "(no subject)"
                header = f"{subject} - {item['from']}"
                with st.expander(header):
                    st.markdown(f"**Date:** {item['date']}")
                    if item["body"]:
                        st.text(item["body"])
                    else:
                        st.caption("No plain-text body.")
                    if item["attachments"]:
                        st.markdown("**Saved attachments:**")
                        for path in item["attachments"]:
                            st.code(path)
                    else:
                        st.caption("No attachments saved.")

    st.subheader("AI Test Detection")
    prompt_template = st.text_area("Prompt", value=DEFAULT_PROMPT, height=260)

    # Optional: show status (no key entry field)
    if not get_openai_api_key():
        st.warning("Missing OpenAI API key in Streamlit secrets (OPENAI_API_KEY).")

    pdf_paths = []
    if source_mode == "Fetch from email" and save_attachments_enabled:
        pdf_paths = sorted(attachments_root.rglob("*.pdf"))

    if source_mode == "Upload PDFs":
        available_count = len(uploads) if uploads else 0
    else:
        available_count = len(pdf_paths)

    st.caption(f"Ready to analyze {available_count} PDF file(s).")
    run_clicked = st.button("Run AI detection")

    if run_clicked:
        api_key = get_openai_api_key()
        if not api_key:
            st.error("Missing OpenAI API key in Streamlit secrets (OPENAI_API_KEY).")
            st.stop()

        client = OpenAI(api_key=api_key, http_client=http_client)

        progress = st.progress(0)
        current_file = st.empty()
        results = []
        total = available_count or 1
        processed = 0

        if source_mode == "Upload PDFs":
            if not uploads:
                st.error("No PDF files uploaded.")
                st.stop()
            for file in uploads:
                current_file.caption(f"Analyzing: {file.name}")
                data = file.read()
                text = extract_text_from_pdf(data)
                if not text.strip():
                    results.append({
                        "file_name": file.name,
                        "source": "upload",
                        "file_path": "",
                        "result": "no extractable text",
                        "note": "",
                    })
                    processed += 1
                    progress.progress(min(processed / total, 1.0))
                    continue

                result = llm_analyze(text, prompt_template, client)
                results.append({
                    "file_name": file.name,
                    "source": "upload",
                    "file_path": "",
                    "result": result,
                    "note": "",
                })
                processed += 1
                progress.progress(min(processed / total, 1.0))
        else:
            if not save_attachments_enabled:
                st.error("Enable 'Save attachments' to analyze email PDFs.")
                st.stop()
            if not pdf_paths:
                st.error("No PDF files found in the attachments folder.")
                st.stop()
            for pdf_path in pdf_paths:
                current_file.caption(f"Analyzing: {pdf_path.name}")
                text = extract_text_from_path(pdf_path)
                if not text.strip():
                    results.append({
                        "file_name": pdf_path.name,
                        "source": "email",
                        "file_path": str(pdf_path),
                        "result": "no extractable text",
                        "note": "",
                    })
                    processed += 1
                    progress.progress(min(processed / total, 1.0))
                    continue

                result = llm_analyze(text, prompt_template, client)
                results.append({
                    "file_name": pdf_path.name,
                    "source": "email",
                    "file_path": str(pdf_path),
                    "result": result,
                    "note": "",
                })
                processed += 1
                progress.progress(min(processed / total, 1.0))

        current_file.caption("Analysis complete.")
        st.session_state["ai_results"] = results

    results = st.session_state.get("ai_results")
    if results:
        st.subheader("AI Results")
        display_rows = [
            {
                "file_name": item.get("file_name", ""),
                "result": item.get("result", ""),
            }
            for item in results
        ]
        st.dataframe(display_rows, use_container_width=True)

        csv_buffer = io.StringIO()
        writer = csv.DictWriter(
            csv_buffer,
            fieldnames=["file_name", "result"],
        )
        writer.writeheader()
        writer.writerows(display_rows)

        st.download_button(
            "Export results as CSV",
            data=csv_buffer.getvalue(),
            file_name="ai_results.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
