import requests
from bs4 import BeautifulSoup
import os

BASE_URL = "https://www.gutenberg.org"
SEARCH_URL = BASE_URL + "/ebooks/search/?query=napoleon+bonaparte"
os.makedirs("corpus", exist_ok=True)

def get_full_url(relative_url):
    return BASE_URL + relative_url

def download_text(title, text_url):
    try:
        response = requests.get(text_url)
        if response.status_code == 200:
            filename = f"corpus/{title.replace(' ', '_')}.txt"
            with open(filename, "w", encoding="utf-8") as file:
                file.write(response.text)
            print(f"Downloaded: {title}")
        else:
            print(f"Failed to download {title}: {response.status_code}")
    except Exception as e:
        print(f"Error downloading {title}: {e}")

def scrape_books(page_url):
    response = requests.get(page_url)
    soup = BeautifulSoup(response.text, "html.parser")
    book_links = soup.select("a[href^='/ebooks/']")

    for link in book_links:
        title = link.text.strip()
        relative_url = link["href"]
        book_url = get_full_url(relative_url)

        # Open the book page to find the plain text link
        book_page = requests.get(book_url)
        book_soup = BeautifulSoup(book_page.text, "html.parser")
        text_link = book_soup.find("a", text="Plain Text UTF-8")

        if text_link:
            text_url = get_full_url(text_link["href"])
            download_text(title, text_url)

    # Find the "Next" button for pagination
    next_button = soup.find("a", text="Next")
    if next_button:
        next_url = get_full_url(next_button["href"])
        print(f"Moving to next page: {next_url}")
        scrape_books(next_url)

# Start scraping from the first page
scrape_books(SEARCH_URL)
