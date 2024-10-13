import argparse

import torch


def add_prefix_to_state_dict(checkpoint_path: str, output_path: str, prefix: str) -> None:
    """
    Loads a PyTorch checkpoint, adds a prefix to each key in the state_dict,
    and saves the modified checkpoint.

    Args:
        checkpoint_path (str): Path to the input checkpoint.
        output_path (str): Path where the modified checkpoint will be saved.
        prefix (str): The prefix to add to each key in the state_dict.
    """
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)

    # Check if 'state_dict' is part of the checkpoint
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint  # If checkpoint is just the state_dict

    # Create a new state_dict with the prefix added to each key
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = prefix + key
        new_state_dict[new_key] = value

    # If the checkpoint has additional information (other than state_dict), preserve it
    modified_checkpoint = checkpoint.copy()
    modified_checkpoint["state_dict"] = new_state_dict

    # Save the modified checkpoint
    torch.save(modified_checkpoint, output_path)
    print(f"Modified checkpoint saved to {output_path}")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Add a prefix to each key in the PyTorch state_dict and save the modified checkpoint."
    )
    parser.add_argument("checkpoint_path", type=str, help="Path to the input checkpoint file.")
    parser.add_argument("output_path", type=str, help="Path to save the modified checkpoint file.")
    parser.add_argument("prefix", type=str, help="Prefix to add to each key in the state_dict.")

    args = parser.parse_args()

    # Add prefix to state_dict and save the modified checkpoint
    add_prefix_to_state_dict(args.checkpoint_path, args.output_path, args.prefix)


if __name__ == "__main__":
    main()
