import subprocess
import time
import json
from urllib.parse import quote

f = open("sheet.tsv")

API_KEYS = [
    "AIzaSyDVS6Bq6bGg0_UFWp5BikwERUB_Yr6mkg8",
    "AIzaSyBL89erFI1xMxuWg1QQeyFGv6IILXNUHT8",
    "AIzaSyAsR9kFLNov7TCcTZ5kEQMLOZNRGq7bPi8",
    "AIzaSyDTL7KkJeZLanbtDMFRGUq7qvzurWTrtkM",
    "AIzaSyB_8CofFgnb66L3Q93X47lM-NZN8bZB2v4",
    "AIzaSyDHvHt-KZ6XSYNWkkNWtaU6EFfc806sVHY",
    "AIzaSyAer1ATIijLxePhDd10NAtNFZwxObRJ3B4",
    "AIzaSyCQmoe2h-AJBCKc6sQsTEBuISl6SqpaC5U"
]

i = 0
api_key = 5
while True:
    line = f.readline()
    if not line: break
    line_ = line.rstrip().split('\t')
    artist = line_[0]
    title = line_[1]
    search = quote(line_[0] + " " + line_[1])
    if i >= 2571:
    # print("https://youtube.googleapis.com/youtube/v3/search?part=snippet&maxResults=1&q=" + search + "&key=AIzaSyDVS6Bq6bGg0_UFWp5BikwERUB_Yr6mkg8")
        subprocess.run([
            "curl",
            "https://youtube.googleapis.com/youtube/v3/search?part=snippet&maxResults=1&q=" + search + "&key=" + API_KEYS[api_key],
            "--header",
            "Accept: application/json",
            "--compressed",
            "-o",
            "temp.txt"
        ])
        j = open("temp.txt")
        d = json.loads(j.read())
        j.close()
        try:
            print(d["items"][0]["id"]["videoId"])
            urls = open("urls.txt", "a")
            urls.write(d["items"][0]["id"]["videoId"])
            urls.write("\n")
            urls.close()
        except:
            break
    i += 1
    
f.close()