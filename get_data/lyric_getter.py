from random import random
import subprocess
import re
import time
from bs4 import BeautifulSoup

f = open("data/sheet.tsv")

i = 0

def absolve(test_str : str) -> str:
    ret = ''
    skip1c = 0
    skip2c = 0
    for i in test_str:
        if i == '[':
            skip1c += 1
        elif i == ']' and skip1c > 0:
            skip1c -= 1
        elif skip1c == 0 and skip2c == 0:
            ret += i
    return re.sub(r'[^a-zA-Z\ ]', '', ret).lower().lstrip().rstrip()

def kebab_caser(s : str, capitalize_first = True):
    s = re.sub(r'[^a-z0-9\-]', '', "-".join(s.lower().split(" ")))
    if capitalize_first:
        s = s[0].upper() + s[1:]
    return s


while True:
    line = f.readline()
    if not line: break
    o = open("data/lyrics.txt", "a")
    line_ = line.rstrip().split('\t')
    artist = line_[0]
    title = line_[1]
    search = kebab_caser(artist) + "-" + kebab_caser(title, False) + "-lyrics"
    print("https://genius.com/" + search)
    a = subprocess.call([
        "curl",
        "https://genius.com/" + search,
        "-o",
        "output.html"
    ])
    f_ = open("output.html")
    soup = BeautifulSoup(f_.read(), 'html.parser')
    time.sleep(random() * 5 + 3)
    for container in soup.find_all(class_=re.compile("Lyrics__Container")):
        soup1 = BeautifulSoup(str(container), 'html.parser')
        o.write(absolve(soup1.get_text(" ")) + " ")
    o.write('\n')
    o.close()
    print()
    i += 1
f.close()