import requests
from bs4 import BeautifulSoup

def inspect_page(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    # Print the structure of the page
    for tag in soup.find_all(["a", "p", "h1", "h2", "div"]):
        print(f"Tag: {tag.name}, Text: {tag.get_text(strip=True)[:60]}, Link: {tag.get('href')}")

# Test with Project Gutenberg search results
inspect_page("https://www.gutenberg.org/ebooks/search/?query=napoleon+bonaparte")
