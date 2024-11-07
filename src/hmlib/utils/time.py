def format_duration_to_hhmmss(seconds: float, decimals: int = 4) -> str:
    """
    Convert a duration in seconds to a formatted string in HH:MM:SS.ssss format,
    with customizable precision for the seconds.

    Args:
    seconds (float): The duration in seconds.
    decimals (int): The number of decimal places for the seconds.

    Returns:
    str: The formatted duration as a string.
    """
    # Calculate hours, minutes, and the remaining seconds
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remaining_seconds = seconds % 60

    # Determine the format string for seconds based on the specified decimal places
    seconds_format = f"{{:06.{decimals}f}}"

    # Format the duration to HH:MM:SS.ssss
    formatted_duration = f"{hours:02}:{minutes:02}:{seconds_format.format(remaining_seconds)}"

    return formatted_duration
