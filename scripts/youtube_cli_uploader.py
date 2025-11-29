#!/usr/bin/env python3
"""
Simple YouTube CLI uploader.

Requirements:
    pip install --upgrade google-api-python-client google-auth-oauthlib google-auth-httplib2
    sudo apt install ffmpeg   # only needed for --test

Before first use:
    1. Create a Google Cloud project.
    2. Enable "YouTube Data API v3".
    3. Create OAuth 2.0 client ID (Desktop or Web application) and download the JSON.
    4. Save it as "client_secrets.json" next to this script (or pass --client-secrets).

Usage examples:

    # Upload a real file
    python youtube_cli_uploader.py myvideo.mp4 \
        --title "My video" \
        --description "Uploaded from CLI" \
        --tags hockey cli test \
        --privacy unlisted \
        --playlist-id PLxxxxxxxxxxxxxxxx

    # Test upload (creates a tiny 2s test clip and uploads it)
    python youtube_cli_uploader.py --test
"""

from __future__ import annotations

import argparse
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

# Scopes:
# - youtube.upload: upload videos
# - youtube: manage YouTube account (needed for adding to playlists)
SCOPES = [
    "https://www.googleapis.com/auth/youtube",
    "https://www.googleapis.com/auth/youtube.upload",
]


def get_default_paths() -> Tuple[Path, Path]:
    script_dir = Path(__file__).resolve().parent
    client_secrets = script_dir / "client_secrets.json"
    token_file = script_dir / "youtube_token.json"
    return client_secrets, token_file


def get_credentials(client_secrets_file: Path, token_file: Path) -> Credentials:
    """Load or obtain OAuth2 credentials."""
    creds: Optional[Credentials] = None

    if token_file.exists():
        creds = Credentials.from_authorized_user_file(str(token_file), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            # Try to refresh existing credentials
            creds.refresh(Request())
        else:
            if not client_secrets_file.exists():
                raise FileNotFoundError(
                    f"Client secrets file not found: {client_secrets_file}\n"
                    "Download it from Google Cloud Console and save it there, "
                    "or pass --client-secrets to this script."
                )
            # Run local server flow to get new credentials
            flow = InstalledAppFlow.from_client_secrets_file(str(client_secrets_file), SCOPES)
            creds = flow.run_local_server(port=8080, prompt="consent")

        # Save the credentials for next time
        token_file.write_text(creds.to_json(), encoding="utf-8")

    return creds


def build_youtube_service(creds: Credentials):
    """Build a YouTube Data API client."""
    return build("youtube", "v3", credentials=creds)


def upload_video(
    youtube,
    file_path: Path,
    title: str,
    description: str,
    category_id: str,
    tags: Optional[List[str]],
    privacy_status: str,
) -> str:
    """
    Upload a video file to YouTube.
    Returns the video ID.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Video file not found: {file_path}")

    body = {
        "snippet": {
            "title": title,
            "description": description,
            "categoryId": category_id,
        },
        "status": {
            "privacyStatus": privacy_status,
        },
    }

    if tags:
        body["snippet"]["tags"] = tags

    media = MediaFileUpload(
        str(file_path),
        mimetype="video/*",
        chunksize=-1,  # upload in a single chunk; set a number for resumable
        resumable=True,
    )

    request = youtube.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media,
    )

    print(f"Starting upload: {file_path}")
    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            pct = int(status.progress() * 100)
            print(f"Upload progress: {pct}%")
    video_id = response["id"]
    print(f"Upload complete. Video ID: {video_id}")
    print(f"Video URL: https://www.youtube.com/watch?v={video_id}")
    return video_id


def add_to_playlist(youtube, playlist_id: str, video_id: str) -> None:
    """Add an uploaded video to a playlist."""
    print(f"Adding video {video_id} to playlist {playlist_id}...")
    youtube.playlistItems().insert(
        part="snippet",
        body={
            "snippet": {
                "playlistId": playlist_id,
                "resourceId": {
                    "kind": "youtube#video",
                    "videoId": video_id,
                },
            }
        },
    ).execute()
    print("Playlist update complete.")


def create_test_video() -> Tuple[Path, str, str]:
    """
    Create a tiny 2-second test video using ffmpeg and return (path, title, description).
    Requires ffmpeg to be installed.
    """
    tmp_dir = Path(tempfile.gettempdir())
    out_path = tmp_dir / "youtube_cli_test_upload.mp4"
    title = "CLI test upload"
    description = "Small test clip uploaded by youtube_cli_uploader.py (test mode)."

    if out_path.exists():
        return out_path, title, description

    print(f"Creating test video at {out_path} (requires ffmpeg)...")
    cmd = [
        "ffmpeg",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=640x360:rate=25",
        "-t",
        "2",
        "-pix_fmt",
        "yuv420p",
        "-y",
        str(out_path),
    ]

    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found. Install it with 'sudo apt install ffmpeg' " "or provide your own small test video file."
        )

    return out_path, title, description


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    default_client_secrets, default_token_file = get_default_paths()

    parser = argparse.ArgumentParser(description="Upload a video to your YouTube channel from the command line.")
    parser.add_argument(
        "file",
        nargs="?",
        help="Path to the video file to upload (omit when using --test).",
    )
    parser.add_argument(
        "--title",
        help="Video title (defaults to file name without extension, or a test title for --test).",
    )
    parser.add_argument(
        "--description",
        default="",
        help="Video description.",
    )
    parser.add_argument(
        "--tags",
        nargs="*",
        help="Space-separated list of tags, e.g.: --tags hockey cli test",
    )
    parser.add_argument(
        "--privacy",
        choices=["public", "unlisted", "private"],
        default="unlisted",
        help="Privacy status for the uploaded video (default: unlisted).",
    )
    parser.add_argument(
        "--category-id",
        default="22",
        help="YouTube category ID (default: 22 = People & Blogs).",
    )
    parser.add_argument(
        "--playlist-id",
        help="Optional playlist ID to add the uploaded video to.",
    )
    parser.add_argument(
        "--client-secrets",
        type=Path,
        default=default_client_secrets,
        help=f"Path to OAuth2 client secrets JSON (default: {default_client_secrets}).",
    )
    parser.add_argument(
        "--token-file",
        type=Path,
        default=default_token_file,
        help=f"Path to store OAuth token JSON (default: {default_token_file}).",
    )
    parser.add_argument(
        "--test",
        action="toggle",
        help="Create and upload a small built-in test video (ignores positional FILE).",
    )

    args = parser.parse_args(argv)

    if not args.test and not args.file:
        parser.error("You must specify a video FILE or use --test.")

    if args.test and args.file:
        parser.error("Do not provide a FILE when using --test.")

    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    try:
        creds = get_credentials(args.client_secrets, args.token_file)
        youtube = build_youtube_service(creds)

        if args.test:
            video_path, default_title, default_desc = create_test_video()
            title = args.title or default_title
            description = args.description or default_desc
        else:
            video_path = Path(args.file).expanduser().resolve()
            if not video_path.exists():
                print(f"Error: file not found: {video_path}", file=sys.stderr)
                return 1
            title = args.title or video_path.stem
            description = args.description

        video_id = upload_video(
            youtube=youtube,
            file_path=video_path,
            title=title,
            description=description,
            category_id=args.category_id,
            tags=args.tags,
            privacy_status=args.privacy,
        )

        if args.playlist_id:
            add_to_playlist(youtube, args.playlist_id, video_id)

        return 0

    except HttpError as e:
        print(f"HTTP error during upload: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
