def format_duration_to_hhmmss(seconds: float, decimals: int = 4) -> str:
    """Convert a duration to an ``HH:MM:SS.ssss``-style string.

    @param seconds: Duration in seconds.
    @param decimals: Number of decimal places for the seconds field.
    @return: The formatted duration as a string.
    """
    # Calculate hours, minutes, and the remaining seconds
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remaining_seconds = seconds % 60

    width: int = (3 + decimals) if decimals else 2
    formatted_seconds = f"{remaining_seconds:0{width}.{decimals}f}"

    # Format the duration to HH:MM:SS.ssss
    formatted_duration = f"{hours:02}:{minutes:02}:{formatted_seconds}"

    return formatted_duration


def hhmmss_to_duration_seconds(time_str: str) -> float:
    """Parse an ``HH:MM:SS(.sss)`` or ``MM:SS(.sss)`` string into seconds.

    @param time_str: Duration string such as ``\"01:23:45.6\"`` or ``\"3:21\"``.
    @return: Total duration in seconds as a float.
    @see @ref format_duration_to_hhmmss "format_duration_to_hhmmss" for the inverse.
    """
    # Split the time duration string into components
    h = 0
    m = 0
    s = 0
    tokens = time_str.split(":")
    s = float(tokens[-1])
    if len(tokens) > 1:
        m = int(tokens[-2])
        if len(tokens) > 2:
            assert len(tokens) == 3
            h = int(tokens[0])
    # Extract seconds and milliseconds
    # Convert hours, minutes, seconds, and milliseconds to total seconds
    total_seconds = h * 3600 + m * 60 + s
    return total_seconds
