import os
from typing import Dict

from PIL import Image, ImageDraw, ImageFont

from hmlib.config import ROOT_DIR, load_config_file_yaml
from hmlib.ui.show import show_image
from hmlib.utils.pt_visualization import find_font_path


def create_matchup_image(
    teams_map: Dict[str, str], team1: str, team2: str, text_color: str, bg_color: str
) -> Image.Image:
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
    icon_size = 256
    team_icon_size = 120
    pad_size = 10
    text_size = 24

    img = Image.new("RGBA", (icon_size, icon_size), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Load logos
    logo1 = Image.open(teams_map[team1])
    logo2 = Image.open(teams_map[team2])

    # Resize logos to fit the image
    logo1 = logo1.resize((team_icon_size, team_icon_size))
    logo2 = logo2.resize((team_icon_size, team_icon_size))

    # Place the first logo and team name in the top left
    img.paste(logo1, (pad_size, pad_size))
    draw.text((pad_size, int(team_icon_size + pad_size * 1.5)), team1, fill=text_color)
    # Place the second logo and team name in the bottom right
    team2_y = icon_size - team_icon_size - pad_size
    img.paste(logo2, (icon_size - team_icon_size - pad_size, team2_y))
    text_width, text_height = draw.textsize(team2)
    draw.text(
        (
            icon_size - text_width - text_size,
            team2_y - text_height - pad_size,
        ),
        team2,
        fill=text_color,
    )

    font_path = find_font_path("arial.ttf")
    if not font_path:
        font_path = find_font_path()
    # Add "VS" text in the middle
    try:
        font = ImageFont.truetype(font_path, size=text_size)  # Path to a ttf file might be needed
    except IOError:
        font = ImageFont.load_default()
    draw.text(
        (int(icon_size / 2), int(icon_size / 2)), "VS", fill=text_color, font=font, anchor="mm"
    )

    return img


if __name__ == "__main__":
    icon_image = create_matchup_image(
        teams_map={
            "Sharks": "/home/colivier/src/hm/resources/teams/sharks.png",
            "BlackStars": "/home/colivier/src/hm/resources/teams/blackstars.png",
        },
        team1="Sharks",
        team2="BlackStars",
        text_color="BLUE",
        bg_color="GRAY",
    )
    show_image("Icon", icon_image)
