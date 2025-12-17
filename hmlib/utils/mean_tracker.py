import torch

from hmlib.log import get_logger


logger = get_logger(__name__)


class MeanTracker:
    """Track per-frame tensor means against values stored on disk.

    In ``\"output\"`` mode the tracker appends the mean (or integer sum) of
    each incoming tensor to a text file; in ``\"input\"`` mode it reads the
    file and asserts that subsequent tensors match the recorded values.

    @param file_path: Path to the text file containing one scalar per line.
    @param mode: Either ``\"output\"`` (record) or ``\"input\"`` (verify).
    @see @ref hmlib.utils.video.load_first_video_frame "load_first_video_frame"
         for a simple producer of tensors to validate.
    """

    def __init__(self, file_path: str, mode="output") -> None:
        self.file_path = file_path
        self.mode = mode
        if self.mode == "output":
            # Open the file in append mode to record means
            self.file = open(self.file_path, "a")
        elif self.mode == "input":
            # Read the file to collect all expected means for later comparison
            with open(self.file_path, "r") as f:
                self.expected_means = [float(line.strip()) for line in f]
            self.current_frame = 0
        else:
            raise ValueError("Mode must be either 'output' or 'input'")

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return self.forward(*args, **kwargs)

    @staticmethod
    def get_frame_value(tensor: torch.Tensor) -> torch.Tensor:
        if torch.is_floating_point(tensor):
            return torch.mean(tensor)
        else:
            tensor_sum = torch.sum(tensor)
            assert tensor_sum.dtype == torch.int64
            return tensor_sum

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.mode == "output":
            # Compute the mean of the tensor and write to the file
            mean_value = self.get_frame_value(tensor).item()
            self.file.write(f"{mean_value}\n")
        elif self.mode == "input":
            # Compute the mean of the tensor and compare it to the expected mean
            if self.current_frame >= len(self.expected_means):
                raise IndexError("No more expected means to compare.")
            mean_value = self.get_frame_value(tensor).item()
            expected_mean = self.expected_means[self.current_frame]
            if not torch.isclose(torch.tensor(mean_value), torch.tensor(expected_mean), atol=1e-6):
                logger.error(
                    "Mismatch at frame %d: expected %f, got %f, frame %d",
                    self.current_frame,
                    expected_mean,
                    mean_value,
                    self.current_frame,
                )
                assert False
            self.current_frame += 1
        return tensor

    def close(self):
        # Close the file if it's open
        if self.mode == "output" and self.file:
            self.file.close()
        if self.current_frame and self.mode == "input":
            logger.info("Successfully verified %d frames.", self.current_frame)

    def __delete__(self):
        if hasattr(self, "close"):
            self.close()
