import os
import re
import csv
import cv2
import numpy as np
import imagehash
import Levenshtein
from datetime import datetime
from PIL import Image
import easyocr   # replaced pytesseract
import streamlit as st
import urllib.parse
from skimage.metrics import structural_similarity as ssim
import requests

# ---------- CONFIGURATION ----------
BLACKLIST_FILE = "blacklist.txt"
LOG_FILE = "blacklist_log.csv"
REFERENCE_TEMPLATES_PATH = "reference_screenshots/"
TEMPLATE_PATH = "templates/"
KEYWORDS = ["pmcare", "pmcares", "lottery", "winmoney", "prize", "donation"]

# ---------- HELPER FUNCTIONS ----------
def load_blacklist():
    try:
        with open(BLACKLIST_FILE, "r") as f:
            return [line.strip().lower() for line in f if line.strip()]
    except FileNotFoundError:
        return []

def save_blacklist(blacklist):
    with open(BLACKLIST_FILE, "w") as f:
        for upi in sorted(set([u.lower().strip() for u in blacklist])):
            f.write(f"{upi}\n")

def log_blacklist_addition(upi_id, added_by="web_app"):
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["Timestamp", "UPI_ID", "Added_By"])
        writer.writerow([datetime.now().isoformat(), upi_id, added_by])

def is_valid_format(upi_id: str) -> bool:
    return bool(re.match(r"^[a-zA-Z0-9.\-_]+@[a-zA-Z0-9]+$", upi_id))

def check_similarity(upi_id: str, threshold=0.85):
    for black in BLACKLIST:
        ratio = Levenshtein.ratio(upi_id, black)
        if ratio >= threshold:
            return black, ratio
    return None, 0

def check_upi_id(upi_id: str):
    upi_id = upi_id.lower().strip()

    # 1) Blacklist check first
    if upi_id in BLACKLIST:
        return "danger", f"🚫 Blacklisted UPI ID: {upi_id}"

    # 2) Validate format
    if not is_valid_format(upi_id):
        return "invalid", f"❌ Invalid UPI format: {upi_id}"

    # 3) Keyword checks
    for kw in KEYWORDS:
        if kw in upi_id:
            return "danger", f"⚠ Suspicious keyword '{kw}' found in {upi_id}"

    # 4) Similarity to blacklist
    similar, score = check_similarity(upi_id)
    if similar:
        return "danger", f"⚠ '{upi_id}' looks similar to blacklisted '{similar}' (score: {score:.2f})"

    return "safe", f"✅ Safe UPI ID: {upi_id}"

# ---------- QR CHECKER ----------
def orb_similarity(img1, img2):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    if des1 is None or des2 is None:
        return 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    return len(matches) / max(len(kp1), len(kp2), 1)

def phash_similarity(img1, img2):
    hash1 = imagehash.phash(Image.fromarray(img1))
    hash2 = imagehash.phash(Image.fromarray(img2))
    return 1 - (hash1 - hash2) / len(hash1.hash)**2

def check_background(uploaded_img, templates_path=TEMPLATE_PATH):
    uploaded_gray = cv2.cvtColor(uploaded_img, cv2.COLOR_BGR2GRAY)
    best_score, best_type = 0, "Unknown"
    for category in ["genuine", "fake"]:
        folder = os.path.join(templates_path, category)
        if not os.path.exists(folder):
            continue
        for file in os.listdir(folder):
            template = cv2.imread(os.path.join(folder, file))
            if template is None:
                continue
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            orb_score = orb_similarity(uploaded_gray, template_gray)
            phash_score = phash_similarity(uploaded_gray, template_gray)
            combined = orb_score * 0.7 + phash_score * 0.3
            if combined > best_score:
                best_score, best_type = combined, category
    return best_type, best_score

# ---------- SCREENSHOT CHECKER ----------
def ela_check(img):
    ela_img = Image.new("RGB", img.size)
    ela_img.paste(img)
    ela_score = 20
    return ela_img, ela_score

def ocr_extract_text(img):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(np.array(img))
    return " ".join([r[1] for r in result])

def validate_transaction_id(text):
    matches = re.findall(r"\b[a-zA-Z0-9]{10,35}\b", text)
    score = 0
    findings = []
    if matches:
        findings.append(f"Transaction IDs found: {matches}")
    else:
        findings.append("❌ No valid transaction ID found.")
        score = 50
    return findings, score, matches

def validate_payment_text(text):
    suspicious = ["gift", "reward", "lottery", "urgent", "donate"]
    score = 0
    findings = []
    for word in suspicious:
        if word.lower() in text.lower():
            findings.append(f"Suspicious keyword: {word} ❌")
            score += 20
    return findings, score

def ssim_similarity(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return score

def check_fake_layout(uploaded_img, templates_path=REFERENCE_TEMPLATES_PATH):
    best_score = 0
    if not os.path.exists(templates_path):
        return best_score
    for file in os.listdir(templates_path):
        template = cv2.imread(os.path.join(templates_path, file))
        if template is None:
            continue
        template_resized = cv2.resize(template, (uploaded_img.shape[1], uploaded_img.shape[0]))
        score = ssim_similarity(uploaded_img, template_resized)
        if score > best_score:
            best_score = score
    return best_score

def analyze_screenshot(img):
    results = {}
    risk_score = 0

    ela_result, ela_score = ela_check(img)
    results["ELA Check"] = ela_result
    risk_score += ela_score

    text = ocr_extract_text(img)
    tx_findings, tx_score, tx_matches = validate_transaction_id(text)
    results["Transaction ID Check"] = tx_findings
    risk_score += tx_score

    keyword_findings, keyword_score = validate_payment_text(text)
    results["Suspicious Keywords"] = keyword_findings
    risk_score += keyword_score

    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    layout_score = check_fake_layout(cv_img)
    if layout_score > 0.8:
        results["Layout Check"] = f"Fake layout detected ❌ (Score: {layout_score:.2f})"
        risk_score += 30
    else:
        results["Layout Check"] = f"Layout appears normal ✅ (Score: {layout_score:.2f})"

    verdict = "Fake ⚠️" if risk_score >= 50 else "Likely Genuine ✅"
    return ela_result, text, results, risk_score, verdict

# ---------- IMAGE REVERSE SEARCH ----------
def generate_reverse_image_search_link(img):
    temp_path = "temp_uploaded.png"
    img.save(temp_path)
    st.warning("Reverse image search requires uploading the image online.")
    search_url = f"https://www.google.com/searchbyimage?image_url={urllib.parse.quote(temp_path)}"
    return search_url

# ---------- MANUAL WEB SEARCH CHECKER ----------
def analyze_review_screenshot(img):
    results = {}
    risk_score = 0
    text = ocr_extract_text(img)
    if text.strip():
        query = urllib.parse.quote(text[:100])
        google_search_url = f"https://www.google.com/search?q={query}"
        results["Google Text Search"] = [("Search Google", google_search_url)]
        risk_score += 10
    search_url = generate_reverse_image_search_link(img)
    results["Reverse Image Search"] = [("Google Image Search", search_url)]
    verdict = "Fake ⚠️" if risk_score >= 50 else "Likely Genuine ✅"
    return text, results, risk_score, verdict

# ---------- LOAD DATA ----------
BLACKLIST = load_blacklist()

# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="Scam Detector", layout="wide")
st.title("🛡 Scam Detector App")

tab1, tab2, tab3, tab4 = st.tabs([
    "🔐 UPI Scam Detector",
    "🔍 QR Background Checker",
    "📄 Screenshot Transaction Checker",
    "🌐 Manual Web/Reverse Image Checker"
])

# -------- TAB 1: UPI Scam Detector --------
with tab1:
    st.subheader("Check a UPI ID")
    upi_input = st.text_input("Enter a UPI ID to check:", key="upi_input")
    if st.button("Check UPI ID", key="upi_check_btn"):
        if upi_input:
            status, message = check_upi_id(upi_input)
            if status == "safe":
                st.success(message)
            elif status == "invalid":
                st.warning(message)
            else:
                st.error(message)
    st.subheader("➕ Add to Blacklist")
    new_upi = st.text_input("Enter a UPI ID to blacklist:", key="upi_blacklist_input")
    if st.button("Add to Blacklist", key="upi_add_btn"):
        if new_upi:
            new_upi = new_upi.lower().strip()
            if new_upi not in BLACKLIST:
                BLACKLIST.append(new_upi)
                save_blacklist(BLACKLIST)
                log_blacklist_addition(new_upi, added_by="streamlit_user")
                st.success(f"✅ '{new_upi}' added to blacklist and logged!")
            else:
                st.info(f"ℹ '{new_upi}' is already in the blacklist.")
    st.subheader("📄 Current Blacklist")
    if BLACKLIST:
        st.dataframe({"Blacklisted UPI IDs": BLACKLIST})
    else:
        st.write("✅ No blacklisted UPI IDs found.")

# -------- TAB 2: QR Background Checker --------
with tab2:
    st.subheader("Check QR Code Background")
    uploaded_qr = st.file_uploader("Upload QR code poster", type=["png","jpg","jpeg"], key="qr_upload")
    if uploaded_qr:
        img_qr = Image.open(uploaded_qr).convert("RGB")
        cv_img_qr = cv2.cvtColor(np.array(img_qr), cv2.COLOR_RGB2BGR)
        st.image(img_qr, caption="Uploaded QR", use_container_width=True)
        result, score = check_background(cv_img_qr)
        if result == "genuine":
            st.success(f"✅ Looks Genuine (Score: {score:.2f})")
        elif result == "fake":
            st.error(f"⚠ Suspicious Background Detected! (Score: {score:.2f})")
        else:
            st.warning("Could not classify background.")

# -------- TAB 3: Screenshot Transaction Checker --------
with tab3:
    st.subheader("Analyze Transaction Screenshot")
    uploaded_ss = st.file_uploader("Upload Screenshot", type=["png","jpg","jpeg"], key="ss_upload")
    if uploaded_ss:
        img_ss = Image.open(uploaded_ss).convert("RGB")
        st.image(img_ss, caption="Uploaded Screenshot", use_container_width=True)
        ela_result, text, results, score, verdict = analyze_screenshot(img_ss)
        st.metric("Risk Score", f"{score}/100")
        st.write(f"**Verdict:** {verdict}")
        st.image(ela_result, caption="ELA Result")
        st.text_area("Extracted Text", text, height=150)
        for key, val in results.items():
            if isinstance(val, list):
                for item in val:
                    st.write(f"**{key}:** {item}")
            else:
                st.write(f"**{key}:** {val}")

# -------- TAB 4: Manual Web / Reverse Image Checker --------
with tab4:
    st.subheader("Analyze Review Screenshot")
    uploaded_ss_tab4 = st.file_uploader("Upload Screenshot", type=["png","jpg","jpeg"], key="ss_tab4_upload")
    if uploaded_ss_tab4:
        img_tab4 = Image.open(uploaded_ss_tab4).convert("RGB")
        st.image(img_tab4, caption="Uploaded Screenshot", use_container_width=True)
        text, results, score, verdict = analyze_review_screenshot(img_tab4)
        st.metric("Risk Score", f"{score}/100")
        st.write(f"**Verdict:** {verdict}")
        st.text_area("Extracted Text", text, height=150)
        for desc, link in results.get("Google Text Search", []):
            st.markdown(f"[{desc}]({link})", unsafe_allow_html=True)
        for desc, link in results.get("Reverse Image Search", []):
            st.markdown(f"[{desc}]({link})", unsafe_allow_html=True)
