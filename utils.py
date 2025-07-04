
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader

def read_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def scrape_url(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()

        soup = BeautifulSoup(res.content, 'html.parser')
        paragraphs = soup.find_all('p')
        article_text = ' '.join([p.get_text() for p in paragraphs if p.get_text(strip=True)])

        if not article_text.strip():
            return "⚠️ No usable text found on the page."

        return article_text

    except Exception as e:
        return f"⚠️ Error fetching URL: {str(e)}"
