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

    width: int = 3 + decimals
    formatted_seconds = f"{remaining_seconds:0{width}.{decimals}f}"

    # Format the duration to HH:MM:SS.ssss
    formatted_duration = f"{hours:02}:{minutes:02}:{formatted_seconds}"

    return formatted_duration
