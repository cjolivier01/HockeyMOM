from typing import Dict, List, Tuple


def find_low_velocity_ranges(
    data: Dict[int, Dict[int, float]],
    min_velocity: float = 3.0,
    min_frames: int = 30,
    min_slow_track_ratio: float = 0.8,
    frame_step: int = 1,
    min_tracks: int = 7,
) -> List[Tuple[int, int]]:
    """
    Identifies ranges of frame_ids where at least 80% of tracking IDs have a velocity of less than 3
    for at least 30 consecutive frames.

    Args:
    data (Dict[int, Dict[int, float]]): Dictionary with frame_id as keys and another dictionary of
                                        {tracking_id: velocity} as values.

    Returns:
    List[Tuple[int, int]]: List of tuples, where each tuple is a range of frame_ids.
    """
    low_velocity_frames = []

    # Identify frames that meet the velocity condition
    for frame_id, tracks in data.items():
        low_velocity_count = sum(1 for velocity in tracks.values() if velocity < min_velocity)
        total_tracks = len(tracks)
        # if total_tracks < 4:
        #     low_velocity_frames.append(frame_id)
        #     continue
        if total_tracks < min_tracks:
            continue
        if low_velocity_count / total_tracks >= min_slow_track_ratio:
            low_velocity_frames.append(frame_id)

    low_velocity_frames.sort()

    # Find continuous ranges of at least 30 frames
    ranges = []
    if low_velocity_frames:
        start = low_velocity_frames[0]
        current = start

        for frame in low_velocity_frames[1:]:
            if frame == current + 1 or frame == current + 2:
                current = frame
            else:
                if current - start + 1 >= min_frames:
                    ranges.append((start, current))
                start = frame
                current = start

        # Check the last range
        if current - start + 1 >= min_frames:
            ranges.append((start, current))

    return ranges


if __name__ == "__main__":
    # Example usage with hypothetical data
    data = {
        1: {1: 2.5, 2: 2.9},
        2: {1: 2.4, 2: 2.8},
        3: {1: 2.3, 2: 2.7},
        4: {1: 2.2, 2: 2.6},
        # Assuming continuity up to frame 35 for demonstration
        **{i: {1: 1.5, 2: 2.0} for i in range(5, 36)},
    }

    ranges = find_low_velocity_ranges(data)
    print("Low velocity ranges:", ranges)
