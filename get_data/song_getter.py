import os
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("start", metavar = "start", type = int)
parser.add_argument("end", metavar = "end", type = int)
args = parser.parse_args()

f = open("get_data/urls.txt")
existing_songs = os.listdir("data/all_2595/")

i = 1

def zfill(i):
    i_ = str(i)
    return "0" * (4 - len(i_)) + i_

while True:
    line = f.readline()
    if not line: break
    if (zfill(i)) + ".wav" not in existing_songs and i >= args.start and i < args.end:
        subprocess.run([
            "youtube-dl",
            "https://www.youtube.com/watch?v=" + line.rstrip(),
            "-x",
            "--audio-format",
            "wav",
            "--audio-quality",
            "0"
        ])
        k = os.listdir(".")
        song_file = ""
        for file in k:
            if file.endswith(".wav"):
                song_file = file
                break
        subprocess.run([
            "ffmpeg",
            "-ss",
            "60",
            "-i",
            song_file,
            "-t",
            "7",
            "-c",
            "copy",
            "data/all_2595/" + zfill(i) + ".wav",
        ])
        subprocess.run([
            "rm",
            song_file
        ])
    i += 1