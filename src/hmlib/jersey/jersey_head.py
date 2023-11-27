from yolox.evaluators.mot_evaluator import TrackingHead


class JerseyHead(TrackingHead):
    def __init__(self):
        pass

    def process_tracking(
        self,
        frame_id,
        online_tlwhs,
        online_ids,
        detections,
        info_imgs,
        letterbox_img,
        inscribed_img,
        original_img,
        online_scores,
    ):
        return online_tlwhs, detections
