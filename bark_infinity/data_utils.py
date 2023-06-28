import requests
import bs4
import json
import multiprocessing
import subprocess
import shutil
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List

HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:104.0) Gecko/20100101 Firefox/104.0"}
BASE_URL = "https://www.101soundboards.com"


def convert_mp3_to_wav(mp3_path: str, wav_path: str) -> None:
    subprocess.run(["ffmpeg", "-i", mp3_path, wav_path])


def find_sounds(url: str) -> List[Dict[str, str]]:
    res = requests.get(url, headers=HEADERS)
    res.raise_for_status()

    soup = bs4.BeautifulSoup(res.text, "html.parser")
    scripts = soup.find_all("script")

    for script in scripts:
        if "board_id" not in str(script):
            continue

        trimmed_script = str(script)[
            str(script).find("board_data_inline") + 20 : str(script).find("}]};") + 3
        ]
        sound_list = json.loads(trimmed_script)
        return [
            {
                "id": sound["id"],
                "title": sound["sound_transcript"],
                "url": sound["sound_file_url"],
                "sound_file_pitch": sound["sound_file_pitch"],
            }
            for sound in sound_list["sounds"]
        ]

    raise ValueError("Could not find sounds at provided URL")


def download_sound(url: str, filepath: str) -> None:
    res = requests.get(BASE_URL + url, headers=HEADERS)
    res.raise_for_status()

    with open(filepath, "wb") as f:
        f.write(res.content)


def handle_sound(sound: Dict[str, str], output_directory: str) -> None:
    sound_file_pitch = str(float(sound["sound_file_pitch"]) / 10)
    original_path = os.path.join(output_directory, f'{sound["title"]}-{sound["id"]}')
    download_sound(sound["url"], original_path)

    try:
        wav_path = f"{original_path}.wav"
        convert_mp3_to_wav(original_path, wav_path)
        os.remove(original_path)
    except Exception as e:
        print(f"Failed to convert file: {original_path}, error: {str(e)}")


def fetch_and_convert_sounds(download_directory: str, soundboard_url: str) -> None:
    if not shutil.which("ffmpeg"):
        raise EnvironmentError("ffmpeg not found. Please install ffmpeg in your system.")

    if os.path.exists(download_directory):
        download_directory += f'_{datetime.now().strftime("%Y%m%d%H%M%S")}'

    Path(download_directory).mkdir(exist_ok=True)
    sounds = find_sounds(soundboard_url)

    with multiprocessing.Pool() as pool:
        pool.starmap(handle_sound, [(sound, download_directory) for sound in sounds])
