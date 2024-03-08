from hmlib.builder import PIPELINES


@PIPELINES.register_module()
class HmImageToTensor:
    """Convert image to :obj:`torch.Tensor` by given keys.

    The dimension order of input image is (H, W, C). The pipeline will convert
    it to (C, H, W). If only 2 dimension (H, W) is given, the output would be
    (1, H, W).

    Args:
        keys (Sequence[str]): Key of images to be converted to Tensor.
    """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        """Call function to convert image in results to :obj:`torch.Tensor` and
        permute the channel order.

        Args:
            results (dict): Result dict contains the image data to convert.

        Returns:
            dict: The result dict contains the image converted
                to :obj:`torch.Tensor` and permuted to (C, H, W) order.
        """
        for key in self.keys:
            img = results[key]
            if not torch.is_floating_point(img):
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                #results[key] = to_tensor(img).permute(2, 0, 1).contiguous()
                assert img.dtype == torch.uint8
                results[key] = img.to(torch.float, non_blocking=True) / 255.0
            else:
                results[key] = img
        return results

    def __repr__(self):
        return self.__class__.__name__ + f"(keys={self.keys})"
