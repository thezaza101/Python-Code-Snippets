#!/usr/bin/python3

"""
    search.py

    MediaWiki Action API Code Samples
    Demo of `Search` module: Search for a text or title
    MIT license
"""

import requests

S = requests.Session()

URL = "https://en.wikipedia.org/w/api.php"

SEARCHPAGE = "ipod"

PARAMS = {
    'action':"query",
    'list':"search",
    'srsearch': SEARCHPAGE,
    'format':"json"
}

R = S.get(url=URL, params=PARAMS)
DATA = R.json()

if DATA['query']['search'][0]['title'] == SEARCHPAGE:
    print("Your search page '" + SEARCHPAGE + "' exists on English Wikipedia")

print(DATA)