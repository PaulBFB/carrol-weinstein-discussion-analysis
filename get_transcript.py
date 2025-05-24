import pathlib

import pandas as pd
from bs4 import BeautifulSoup
from httpx import AsyncClient
import asyncio
import json


async def get_url(url: str) -> BeautifulSoup:
    async with AsyncClient() as client:
        response = await client.get(url)

    return BeautifulSoup(response.content, "html.parser")


def load_html(html_path: pathlib.Path) -> BeautifulSoup:
    with open(html_path, mode="r") as file:
        content = file.read()

    soup = BeautifulSoup(content, features="html.parser")

    return soup


def extract_youtube_transcript(soup: BeautifulSoup):
    """
    Extract timestamps and text from YouTube transcript HTML file.

    Args:
        html_file_path (str): Path to the HTML file

    Returns:
        list: List of dictionaries containing timestamp and text
    """

    # Parse HTML with BeautifulSoup
    # soup = BeautifulSoup(html_content, "html.parser")

    # Find all transcript segment renderers
    transcript_segments = soup.find_all("ytd-transcript-segment-renderer")

    extracted_data = []

    for segment in transcript_segments:
        # Extract timestamp
        timestamp_div = segment.find("div", class_="segment-timestamp")
        if timestamp_div:
            timestamp = timestamp_div.get_text(strip=True)
        else:
            timestamp = None

        # Extract text
        text_element = segment.find("yt-formatted-string", class_="segment-text")
        if text_element:
            text = text_element.get_text(strip=True)
        else:
            text = None

        # Only add if both timestamp and text are found
        if timestamp and text:
            extracted_data.append({"timestamp": timestamp, "text": text})

    return extracted_data


def extract_transcript_with_css_selector(html_file_path):
    """
    Alternative extraction method using CSS selectors more specifically.

    Args:
        html_file_path (str): Path to the HTML file

    Returns:
        list: List of dictionaries containing timestamp and text
    """

    with open(html_file_path, "r", encoding="utf-8") as file:
        html_content = file.read()

    soup = BeautifulSoup(html_content, "html.parser")

    # Try to find the specific panel first
    panels = soup.select("#panels > ytd-engagement-panel-section-list-renderer")

    if len(panels) >= 3:
        # Get the third panel (index 2)
        transcript_panel = panels[2]
        transcript_segments = transcript_panel.find_all(
            "ytd-transcript-segment-renderer"
        )
    else:
        # Fallback to finding all transcript segments
        transcript_segments = soup.find_all("ytd-transcript-segment-renderer")

    extracted_data = []

    for segment in transcript_segments:
        timestamp_div = segment.find("div", class_="segment-timestamp")
        text_element = segment.find("yt-formatted-string", class_="segment-text")

        if timestamp_div and text_element:
            timestamp = timestamp_div.get_text(strip=True)
            text = text_element.get_text(strip=True)

            extracted_data.append({"timestamp": timestamp, "text": text})

    return extracted_data


def save_extract(
    extract: list[dict],
    out_file: pathlib.Path = pathlib.Path.cwd() / "data" / "transcript.csv",
):
    data = pd.DataFrame(extract)

    with open(out_file, mode="w") as file:
        data.to_csv(file, index=False)


if __name__ == "__main__":
    from pprint import pprint

    test_url = "https://www.youtube.com/watch?v=5m7LnLgvMnM"

    file_path = pathlib.Path.cwd() / "data" / "ts.html"
    # res = asyncio.run(get_url(test_url))

    test_extract = extract_transcript_with_css_selector(file_path)
    save_extract(test_extract)

    pprint(test_extract)
