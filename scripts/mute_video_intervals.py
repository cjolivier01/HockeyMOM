"""
Mute vuideo intervals

mute_video_intervals.py <input_video_file> <list_of_intervals_in_seconds> <output_video_file>

sample usage:

    python mute_video_intervals.py input_video.mp4 "10-20,40-50" output_video.mp4

"""
import argparse
import subprocess


def make_parser():
    parser = argparse.ArgumentParser("Mute intervals tool")
    parser.add_argument(
        "input_file",
        help="Input video file name",
    )
    parser.add_argument(
        "intervals",
        help="List of start and stop times for intervals to mute (formatted as 'start1-end1,start2-end2,...')",
    )
    parser.add_argument(
        "output_file",
        help="Output video file name",
    )
    return parser



# def mute_intervals(video_file, intervals, output_file):
#     # Create a complex filter string for ffmpeg to mute the specified intervals
#     filter_complex_str = ""

#     # Generate silencer part of the filter for each interval
#     for i, (start, end) in enumerate(intervals):
#         filter_complex_str += (
#             f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[silent{i}];"
#         )

#     # Create a list of all audio parts, including silent parts
#     audio_parts = "[0:a]"
#     for i, _ in enumerate(intervals):
#         audio_parts += f"[silent{i}]"

#     # Concatenate all audio parts
#     filter_complex_str += f"{audio_parts}concat=n={len(intervals) + 1}:v=0:a=1[outa]"

#     # Build the ffmpeg command
#     command = f'ffmpeg -i {video_file} -filter_complex "{filter_complex_str}" -map 0:v -map "[outa]" {output_file}'

#     print(f"Running command:\n{command}\n")

#     # Execute the command
#     process = subprocess.run(shlex.split(command))

#     # Check if ffmpeg command was successful
#     if process.returncode != 0:
#         print("ffmpeg command failed with error: ", process.stderr)
#     else:
#         print("Video processed successfully. Saved as:", output_file)


# def convert_to_seconds(time_str):
#     """
#     Convert a time string in the format of hh:mm:ss, mm:ss, or ss to a float value of seconds.

#     Args:
#     time_str (str): A string representing time in the format of hh:mm:ss, mm:ss, or ss.

#     Returns:
#     float: The equivalent time in seconds.
#     """
#     # Split the string by ':'
#     parts = time_str.split(":")

#     # Convert the time parts to seconds
#     seconds = 0
#     for part in parts:
#         seconds = seconds * 60 + int(part)

#     return float(seconds)

def parse_time(time_str):
    """Parse time in hh:mm:ss, mm:ss, or ss format and return total seconds."""
    parts = time_str.split(':')
    parts = [int(p) for p in parts]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    elif len(parts) == 2:
        return parts[0] * 60 + parts[1]
    else:
        return parts[0]


# def parse_intervals(intervals_str):
#     # Split the intervals string by comma and then convert each one to a tuple of floats
#     return [
#         tuple(map(float, [convert_to_seconds(i) for i in interval.split("-")]))
#         for interval in intervals_str.split(",")
#     ]


def mute_intervals(audio_file, intervals, output_file):
    """Mute given intervals in an audio file using FFmpeg."""
    filters = []

    print(intervals)

    for interval in intervals:
        start, end = interval.split('-')
        start_sec = parse_time(start)
        end_sec = parse_time(end)
        filters.append(f"volume=enable='between(t,{start_sec},{end_sec})':volume=0")

    filter_str = ','.join(filters)
    command = ['ffmpeg', '-i', audio_file, '-af', filter_str, "-c:v ", "copy", output_file]
    print(" ".join(command))
    subprocess.run(command)


def main(args: argparse.Namespace):
    # Example usage:
    # time_intervals = [(10, 20), (40, 50)]  # Replace with your start and end times in seconds


    mute_intervals(
        args.input_file,
        intervals=args.intervals.split(","),
        output_file=args.output_file,
    )

    # mute_intervals(
    #     args.input_file,
    #     intervals=parse_intervals(args.intervals),
    #     output_file=args.output_file,
    # )


if __name__ == "__main__":
    parser = make_parser()
    #args = make_parser().parse_args()
    args = argparse.Namespace()
    #args.input_file = "tracking_output-with-audio.avi.output_audio.aac"
    args.input_file = "test.avi"
    
    
    args.output_file = "err.avi"
    #args.intervals = "6:21-7:00,30:37-31:18,1:04:42-1:05:12,28:22-29:30,59:28-59:54,9:44-10:06,23:00-24:55,44:26-44:43,27:42-28:02,53:16-53:38"
    args.intervals = "6:21-7:00"
    main(args)
