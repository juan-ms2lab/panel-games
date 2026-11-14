#!/usr/bin/env python3
"""Download trivia from Open Trivia DB (https://opentdb.com, CC BY-SA 4.0) into the game's local data file.

Usage: python scripts/fetch_trivia.py   (takes a few minutes: the API allows one request per 5 seconds)
"""

import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://opentdb.com"
OUT = Path(__file__).resolve().parent.parent / "src" / "panel_games" / "data" / "trivia.json"
# category id -> (name, max questions to keep)
CATEGORIES = {10: ("Books", 200), 11: ("Film", 400), 14: ("Television", 250), 15: ("Video Games", 300),
              17: ("Science & Nature", 400), 18: ("Computers", 250), 19: ("Mathematics", 100),
              20: ("Mythology", 100), 29: ("Comics", 100), 30: ("Gadgets", 60)}
DECKS_BY_CATEGORY = {"nerd": {18, 30, 19, 15}, "engineering": {18, 19, 30}, "movies": {11, 14}}
# Science & Nature (17) only counts as engineering when it is about physics, chemistry or machines.
ENGINEERING_17 = (r"physic|electr|circuit|voltage|current|\bvolt|\bwatt|engine|energy|atom\b|atoms\b|atomic|element|chemical|metal|"
                  r"machine|force|gravity|\blight\b|wave|semiconductor|transistor|magnet|laser|radio|speed|\bmass\b|"
                  r"temperature|\bgas\b|pressure|nuclear|isotope|molecule|\bunit\b|\bsi\b|periodic|alloy|steel|bridge|"
                  r"mechanic|thermo|quantum|relativity|newton|einstein|tesla|\bohm\b|hertz|joule")
KEYWORDS = {
    "scifi": r"star wars|star trek|doctor who|the matrix|\balien\b|aliens\b|terminator|blade runner|\bdune\b|asimov|sci-?fi|"
             r"science fiction|spaceship|starship|\brobots?\b|\bandroids?\b|futurama|hitchhiker|back to the future|avatar|"
             r"interstellar|\bhal 9000|jedi|vulcan|klingon|cyberpunk|time travel|wall-e|mass effect|fallout|stargate|"
             r"battlestar|firefly|red dwarf|cyborg|dystopi|jurassic|half-life 2|portal 2|\bhalo\b|\bgalactic|\bdalek|"
             r"tardis|\bsith\b|skywalker|darth|spock|\bwarp drive|\bbladerunner|\bmatrix\b|\bpredator\b|ex machina",
    "fantasy": r"lord of the rings|the hobbit|tolkien|middle-earth|harry potter|hogwarts|game of thrones|westeros|narnia|"
               r"\bdragons?\b|\bwizards?\b|\bwitcher|elder scrolls|skyrim|morrowind|zelda|dungeons|\belves\b|\belf\b|\borcs?\b|"
               r"sorcer|\bwitch\b|excalibur|\bwarcraft|final fantasy|pratchett|discworld|wheel of time|gandalf|frodo|mordor|"
               r"\bhobbit|\bsauron|\bhogwarts|\bdumbledore|\bsmaug|\bgondor|\bshire\b|\bdnd\b|\bd&d\b|\bpaladin|\bwarlock|"
               r"\bsorcerer|\bunicorn|\bgriffin|\bmerlin\b|\barthurian|\bking arthur|\bcamelot",
}


def get(url: str):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


def decode(s: str) -> str:
    return urllib.parse.unquote(s)


def tag(questions):
    for item in questions:
        tags = {d for d, cats in DECKS_BY_CATEGORY.items() if item["cat"] in cats}
        blob = " ".join([item["q"], item["correct"], *item["wrong"]]).lower()
        if item["cat"] == 17 and re.search(ENGINEERING_17, blob):
            tags.add("engineering")
        for deck, pattern in KEYWORDS.items():
            if re.search(pattern, blob) or (deck == "fantasy" and item["cat"] == 20):
                tags.add(deck)
        item["decks"] = sorted(tags)


def save(questions):
    tag(questions)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"source": "Open Trivia DB (opentdb.com), CC BY-SA 4.0", "questions": questions}, indent=0))
    counts = {d: sum(1 for q in questions if d in q["decks"]) for d in ("nerd", "engineering", "movies", "scifi", "fantasy")}
    print(f"Saved {len(questions)} questions to {OUT}\nDeck sizes: {counts}")


def main():
    if "--retag" in sys.argv:  # re-run deck tagging on the saved file without downloading
        save(json.loads(OUT.read_text())["questions"])
        return
    token = get(f"{API}/api_token.php?command=request")["token"]
    questions, seen = [], set()
    for cat, (name, cap) in CATEGORIES.items():
        got = 0
        while got < cap:
            time.sleep(5.5)
            amount = min(50, cap - got)
            data = get(f"{API}/api.php?amount={amount}&category={cat}&type=multiple&encode=url3986&token={token}")
            if data["response_code"] == 5:  # rate limited
                time.sleep(6)
                continue
            if data["response_code"] != 0:
                print(f"  {name}: stopped (response {data['response_code']}) after {got}")
                break
            for item in data["results"]:
                q = decode(item["question"])
                if q in seen:
                    continue
                seen.add(q)
                questions.append({"q": q, "correct": decode(item["correct_answer"]),
                                  "wrong": [decode(w) for w in item["incorrect_answers"]],
                                  "cat": cat, "difficulty": item["difficulty"]})
                got += 1
            print(f"  {name}: {got}", flush=True)
    save(questions)


if __name__ == "__main__":
    sys.exit(main())
