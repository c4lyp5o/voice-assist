# Language Processing
import re
import time
import webbrowser

def open_website(site):
    webbrowser.open(f"https://{site}.com")
    return f"Opening {site.capitalize()}"

def search_google(query):
    webbrowser.open(f"https://www.google.com/search?q={query}")
    return f"Searching Google for '{query}'"

def tell_time():
    return f"The time is {time.strftime('%I:%M %p')}"

COMMAND_PATTERNS = [
    (re.compile(r"\bopen (youtube|github|google)\b"), lambda m: open_website(m.group(1))),
    (re.compile(r"\bsearch for (.+)"), lambda m: search_google(m.group(1))),
    (re.compile(r"\b(what time is it|tell me the time|the time please|time\??|current time|give me the time|show me the time)\b"), lambda m: tell_time()),
]

def process_command(text):
    text_lc = text.lower()
    for pattern, handler in COMMAND_PATTERNS:
        match = pattern.search(text_lc)
        if match:
            return handler(match)
    return None