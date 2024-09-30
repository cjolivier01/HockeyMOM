from typing import List

from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import Resource, build
from googleapiclient.http import MediaFileUpload


def upload_video_to_youtube(
    video_file_path: str, thumbnail_path: str, title: str, description: str, 
    category_id: int, keywords: List[str], privacy_status: str, playlist_id: str, 
    credentials_file: str) -> str:
    """
    Upload a video to YouTube, set its thumbnail, and add it to a playlist.

    Args:
    - video_file_path (str): Path to the video file.
    - thumbnail_path (str): Path to the thumbnail image file.
    - title (str): Title of the video.
    - description (str): Description of the video.
    - category_id (int): YouTube category ID (e.g., 22 for People & Blogs).
    - keywords (List[str]): List of keywords/tags.
    - privacy_status (str): 'public', 'private', or 'unlisted'.
    - playlist_id (str): ID of the playlist to add the video to.
    - credentials_file (str): Path to the OAuth 2.0 credentials file.

    Returns:
    - str: URL of the uploaded video.
    """
    scopes = ["https://www.googleapis.com/auth/youtube.upload", "https://www.googleapis.com/auth/youtube"]
    flow = InstalledAppFlow.from_client_secrets_file(credentials_file, scopes)
    credentials = flow.run_console()
    youtube = build('youtube', 'v3', credentials=credentials)

    body = {
        'snippet': {
            'title': title,
            'description': description,
            'tags': keywords,
            'categoryId': category_id
        },
        'status': {
            'privacyStatus': privacy_status
        }
    }

    video = MediaFileUpload(video_file_path, chunksize=-1, resumable=True)
    upload_response = youtube.videos().insert(
        part="snippet,status",
        body=body,
        media_body=video
    ).execute()

    youtube.thumbnails().set(
        videoId=upload_response['id'],
        media_body=MediaFileUpload(thumbnail_path)
    ).execute()

    playlist_item_body = {
        'snippet': {
            'playlistId': playlist_id,
            'resourceId': {
                'kind': 'youtube#video',
                'videoId': upload_response['id']
            }
        }
    }
    youtube.playlistItems().insert(
        part="snippet",
        body=playlist_item_body
    ).execute()

    return f"https://www.youtube.com/watch?v={upload_response['id']}"
