from typing import Dict

from PIL import Image, ImageDraw, ImageFont


def create_matchup_image(teams_map: Dict[str, str], team1: str, team2: str, text_color: str, bg_color: str) -> Image.Image:
    """
    Generate a 256x256 image showcasing a matchup between two teams.

    Args:
    - teams_map (Dict[str, str]): Map of team names to their logo image file paths.
    - team1 (str): First team name.
    - team2 (str): Second team name.
    - text_color (str): Color of the text ("VS").
    - bg_color (str): Background color of the image.

    Returns:
    - PIL.Image.Image: The generated image.
    """
    # Create a blank image
    img = Image.new('RGB', (256, 256), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Load logos
    logo1 = Image.open(teams_map[team1])
    logo2 = Image.open(teams_map[team2])

    # Resize logos to fit the image
    logo1 = logo1.resize((64, 64))
    logo2 = logo2.resize((64, 64))

    # Place the first logo and team name in the top left
    img.paste(logo1, (10, 10))
    draw.text((10, 80), team1, fill=text_color)

    # Place the second logo and team name in the bottom right
    img.paste(logo2, (182, 182))
    draw.text((182, 150), team2, fill=text_color)

    # Add "VS" text in the middle
    try:
        font = ImageFont.truetype("arial.ttf", size=24)  # Path to a ttf file might be needed
    except IOError:
        font = ImageFont.load_default()
    draw.text((128, 120), "VS", fill=text_color, font=font, anchor="mm")

    return img
