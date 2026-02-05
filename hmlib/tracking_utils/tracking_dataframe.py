import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from mmengine.structures import InstanceData

from hmlib.bbox.box_functions import convert_tlbr_to_tlwh, tlwh_to_tlbr_multiple
from hmlib.datasets.dataframe import HmDataFrameBase, dataclass_to_json, json_to_dataclass
from hmlib.jersey.number_classifier import TrackJerseyInfo
from hmlib.tracking_utils.utils import get_track_mask

try:
    from mmdet.structures import DetDataSample, TrackDataSample
except Exception:  # pragma: no cover
    DetDataSample = None  # type: ignore
    TrackDataSample = None  # type: ignore

if TYPE_CHECKING:
    from mmdet.structures import DetDataSample as _DetDataSample
    from mmdet.structures import TrackDataSample as _TrackDataSample
else:
    _TrackDataSample = Any  # type: ignore
    _DetDataSample = Any  # type: ignore


class TrackingDataFrame(HmDataFrameBase):
    def __init__(self, *args, input_batch_size: int, **kwargs):
        fields = [
            "Frame",
            "ID",
            "BBox_X",
            "BBox_Y",
            "BBox_W",
            "BBox_H",
            "Scores",
            "Labels",
            "Visibility",
            "JerseyInfo",
            # Optional per-track action annotations (if available)
            "ActionLabel",
            "ActionScore",
            "ActionIndex",
        ]
        super().__init__(*args, fields=fields, input_batch_size=input_batch_size, **kwargs)

    def read_data(self) -> None:
        """Read tracking CSVs supporting both legacy and current schemas."""
        if not self.input_file:
            return
        from hmlib.log import logger

        if not os.path.exists(self.input_file):
            logger.error("Could not open dataframe file: %s", self.input_file)
            self.data = None
            return

        df = pd.read_csv(self.input_file, header=None)
        n = int(df.shape[1])

        base_cols = [
            "Frame",
            "ID",
            "BBox_X",
            "BBox_Y",
            "BBox_W",
            "BBox_H",
            "Scores",
            "Labels",
            "Visibility",
            "JerseyInfo",
        ]
        action_cols = ["ActionLabel", "ActionScore", "ActionIndex"]

        if n == len(base_cols) + 1 + len(action_cols):
            # Legacy: base + PoseIndex + actions
            df.columns = base_cols + ["PoseIndex"] + action_cols
            df = df.drop(columns=["PoseIndex"])
        elif n == len(base_cols) + len(action_cols):
            # Current: base + actions
            df.columns = base_cols + action_cols
        elif n == len(base_cols) + 1:
            # Legacy: base + PoseIndex
            df.columns = base_cols + ["PoseIndex"]
            df = df.drop(columns=["PoseIndex"])
        elif n == len(base_cols):
            # Current: base only
            df.columns = base_cols
        elif n > len(base_cols) + 1 + len(action_cols):
            # Future-proof: allow extra columns; assume legacy layout and ignore extras.
            extra = [f"Extra{i}" for i in range(n - (len(base_cols) + 1 + len(action_cols)))]
            df.columns = base_cols + ["PoseIndex"] + action_cols + extra
            df = df.drop(columns=["PoseIndex"])
        elif n > len(base_cols):
            # Partial action columns (no PoseIndex).
            df.columns = base_cols + action_cols[: n - len(base_cols)]
        else:
            # Pad missing base columns.
            for _ in range(len(base_cols) - n):
                df[df.shape[1]] = np.nan
            df.columns = base_cols

        self.data = df.reindex(columns=self.fields)

    def add_frame_records(
        self,
        frame_id: int,
        tracking_ids: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray,
        jersey_info: List[TrackJerseyInfo] = None,
        tlbr: Optional[np.ndarray] = None,
        tlwh: Optional[np.ndarray] = None,
        action_info: Optional[List[Dict[str, Any]]] = None,
    ):
        if tlwh is None:
            assert tlbr is not None
            tlwh = convert_tlbr_to_tlwh(tlbr)

        frame_id = int(frame_id)
        assert frame_id  # frame id's start at 1
        tracking_ids = self._make_array(tracking_ids)
        tlwh = self._make_array(tlwh)
        scores = self._make_array(scores)
        labels = self._make_array(labels)
        jersey_dict: Dict[int, TrackJerseyInfo] = {}
        if jersey_info is not None:
            for j_info in jersey_info:
                j_t_id = j_info.tracking_id
                # assert j_t_id not in jersey_dict
                if j_t_id in jersey_dict:
                    # why does this happen?
                    from hmlib.log import get_logger

                    get_logger(__name__).info(
                        "Ignoring duplicate jersey tracking id %s", jersey_dict
                    )
                jersey_dict[j_t_id] = dataclass_to_json(j_info)

        def _jersey_item(id: int) -> str:
            v = jersey_dict.get(id)
            if v is None:
                return "{}"
            return v

        # Prepare optional action annotations (string label, float score, int index)
        action_label_map: Dict[int, str] = {}
        action_score_map: Dict[int, float] = {}
        action_index_map: Dict[int, int] = {}
        if action_info is not None:
            try:
                for a in action_info:
                    tid = int(a.get("tracking_id", -1))
                    if tid < 0:
                        continue
                    action_label_map[tid] = str(a.get("label", ""))
                    # Coerce to python floats/ints for pandas
                    action_score_map[tid] = float(a.get("score", 0.0))
                    action_index_map[tid] = int(a.get("label_index", -1))
            except Exception:
                pass

        def _action_label_item(id: int) -> str:
            return action_label_map.get(id, "")

        def _action_score_item(id: int) -> float:
            return float(action_score_map.get(id, 0.0))

        def _action_index_item(id: int) -> int:
            return int(action_index_map.get(id, -1))

        new_record = pd.DataFrame(
            {
                "Frame": [frame_id for _ in range(len(tracking_ids))],
                "ID": tracking_ids,
                "BBox_X": tlwh[:, 0],
                "BBox_Y": tlwh[:, 1],
                "BBox_W": tlwh[:, 2],
                "BBox_H": tlwh[:, 3],
                "Scores": scores,
                "Labels": labels,
                "Visibility": [-1 for _ in range(len(tracking_ids))],
                "JerseyInfo": [_jersey_item(t_id) for t_id in tracking_ids],
                "ActionLabel": [_action_label_item(t_id) for t_id in tracking_ids],
                "ActionScore": [_action_score_item(t_id) for t_id in tracking_ids],
                "ActionIndex": [_action_index_item(t_id) for t_id in tracking_ids],
            }
        )
        self._dataframe_list.append(new_record)
        self.counter += 1

        if self.counter >= self.write_interval:
            self.write_data(self.output_file)
            self.first_write = False
            self.counter = 0  # Reset the counter after writing

    def __getitem__(self, idx: int) -> Optional[_TrackDataSample]:
        # Frame id's start at 1; return a TrackDataSample for this frame if possible
        return self.get_sample_by_frame(frame_id=idx + 1)

    def get_data_by_frame(self, frame_number: int) -> Union[Dict[str, Any], None]:
        """Get all tracking data for a specific frame."""
        if not self.data.empty:
            return self.data[self.data["Frame"] == frame_number]
        else:
            from hmlib.log import get_logger

            get_logger(__name__).warning("No data loaded.")
            return None

    def get_data_dict_by_frame(self, frame_id: int) -> Dict[str, Any]:
        frame_id = int(frame_id)
        # Filter the DataFrame for the specified frame
        frame_data = self.data[self.data["Frame"] == frame_id]
        # Extract columns as NumPy arrays
        tracking_ids = frame_data["ID"].to_numpy()
        scores = frame_data["Scores"].to_numpy()
        labels = frame_data["Labels"].to_numpy()
        tlwh = frame_data[["BBox_X", "BBox_Y", "BBox_W", "BBox_H"]].to_numpy()
        jersey_info = frame_data["JerseyInfo"]

        all_track_jersey_info: List[Optional[TrackJerseyInfo]] = []
        for tid, jersey in zip(tracking_ids, jersey_info):
            obj: Optional[TrackJerseyInfo] = None
            if isinstance(jersey, TrackJerseyInfo):
                obj = jersey
            elif isinstance(jersey, str):
                if jersey and jersey != "{}":
                    try:
                        obj = json_to_dataclass(jersey, TrackJerseyInfo)
                    except Exception:
                        obj = None
            # Ignore invalid or placeholder entries.
            if obj is not None and getattr(obj, "tracking_id", -1) < 0:
                obj = None
            all_track_jersey_info.append(obj)

        return dict(
            frame_id=frame_id,
            tracking_ids=tracking_ids,
            scores=scores,
            bboxes=tlwh,
            labels=labels,
            jersey_info=all_track_jersey_info,
        )

    def add_frame_sample(
        self,
        frame_id: int,
        data_sample: Any,
        jersey_info: Optional[List[TrackJerseyInfo]] = None,
        action_info: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Persist a tracking result provided as a per-frame DetDataSample.

        Expects ``pred_track_instances`` on the provided sample.
        """
        ds = data_sample
        inst = getattr(ds, "pred_track_instances", None)
        if inst is None:
            # Empty frame
            self.add_frame_records(
                frame_id=int(frame_id),
                tracking_ids=np.empty((0,), dtype=np.int64),
                tlbr=np.empty((0, 4), dtype=np.float32),
                scores=np.empty((0,), dtype=np.float32),
                labels=np.empty((0,), dtype=np.int64),
                jersey_info=None,
                action_info=None,
            )
            return
        tids = getattr(inst, "instances_id", np.empty((0,), dtype=np.int64))
        tlbr = getattr(inst, "bboxes", np.empty((0, 4), dtype=np.float32))
        scores = getattr(inst, "scores", np.empty((0,), dtype=np.float32))
        labels = getattr(inst, "labels", np.empty((0,), dtype=np.int64))
        mask = get_track_mask(inst)
        if isinstance(mask, torch.Tensor):
            if isinstance(tids, torch.Tensor):
                tids = tids[mask]
            else:
                mask_np = mask.detach().cpu().numpy()
                tids = np.asarray(tids)[mask_np]
            if isinstance(tlbr, torch.Tensor):
                tlbr = tlbr[mask]
            else:
                mask_np = mask.detach().cpu().numpy()
                tlbr = np.asarray(tlbr)[mask_np]
            if isinstance(scores, torch.Tensor):
                scores = scores[mask]
            else:
                mask_np = mask.detach().cpu().numpy()
                scores = np.asarray(scores)[mask_np]
            if isinstance(labels, torch.Tensor):
                labels = labels[mask]
            else:
                mask_np = mask.detach().cpu().numpy()
                labels = np.asarray(labels)[mask_np]
        self.add_frame_records(
            frame_id=int(frame_id),
            tracking_ids=tids,
            tlbr=tlbr,
            scores=scores,
            labels=labels,
            jersey_info=jersey_info,
            action_info=action_info,
        )

    def get_sample_by_frame(self, frame_id: int) -> Optional[_TrackDataSample]:
        """Reconstruct a TrackDataSample (length=1) for the given frame."""
        rec = self.get_data_dict_by_frame(frame_id)
        if not rec:
            return None
        # Convert TLWH to TLBR for pred_track_instances
        tlwh = rec.get("bboxes", np.empty((0, 4), dtype=np.float32))
        if isinstance(tlwh, np.ndarray):
            tlwh_t = torch.as_tensor(tlwh)
            tlbr_t = tlwh_to_tlbr_multiple(tlwh_t)
        else:
            tlbr_t = tlwh_to_tlbr_multiple(tlwh)
        inst = InstanceData(
            instances_id=torch.as_tensor(rec.get("tracking_ids", np.empty((0,), dtype=np.int64))),
            bboxes=tlbr_t,
            scores=torch.as_tensor(rec.get("scores", np.empty((0,), dtype=np.float32))),
            labels=torch.as_tensor(rec.get("labels", np.empty((0,), dtype=np.int64))),
        )

        # Build a one-frame DetDataSample to wrap pred_track_instances
        if DetDataSample is None or TrackDataSample is None:
            return None
        img_ds = DetDataSample()
        img_ds.pred_track_instances = inst
        try:
            # Attach auxiliary info into metainfo
            meta: Dict[str, Any] = {"frame_id": int(frame_id)}
            # jersey info as json-serializable list of dicts
            jerseys: List[Optional[TrackJerseyInfo]] = rec.get("jersey_info", [])
            if jerseys:
                meta["jersey_info"] = [
                    dataclass_to_json(j) if j is not None else None for j in jerseys
                ]
            # action info reconstructed from columns if present
            try:
                df = self.data[self.data["Frame"] == int(frame_id)]
                if not df.empty and "ActionIndex" in df.columns:
                    actions: List[Dict[str, Any]] = []
                    for _, row in df.iterrows():
                        tid = int(row.get("ID", -1))
                        if tid < 0:
                            continue
                        actions.append(
                            dict(
                                tracking_id=tid,
                                label=str(row.get("ActionLabel", "")),
                                label_index=int(row.get("ActionIndex", -1)),
                                score=float(row.get("ActionScore", 0.0)),
                            )
                        )
                    meta["action_results"] = actions
            except Exception:
                pass
            img_ds.set_metainfo(meta)
        except Exception:
            pass

        vds = TrackDataSample()
        vds.video_data_samples = [img_ds]
        try:
            vds.set_metainfo({"key_frames_inds": [0]})
        except Exception:
            pass
        return vds

    def get_samples(
        self, start_frame: Optional[int] = None, end_frame: Optional[int] = None
    ) -> Optional[_TrackDataSample]:
        """Reconstruct a multi-frame TrackDataSample for a frame range (inclusive)."""
        if self.data is None or self.data.empty or TrackDataSample is None or DetDataSample is None:
            return None
        frames = sorted(set(int(f) for f in self.data["Frame"].tolist()))
        if not frames:
            return None
        if start_frame is not None or end_frame is not None:
            lo = int(start_frame) if start_frame is not None else frames[0]
            hi = int(end_frame) if end_frame is not None else frames[-1]
            frames = [f for f in frames if lo <= f <= hi]
        video_samples: List[_DetDataSample] = []
        for f in frames:
            ts = self.get_sample_by_frame(f)
            if ts is None:
                # Empty per-frame container
                inst = InstanceData(
                    instances_id=torch.empty((0,), dtype=torch.long),
                    bboxes=torch.empty((0, 4), dtype=torch.float32),
                    scores=torch.empty((0,), dtype=torch.float32),
                    labels=torch.empty((0,), dtype=torch.long),
                )
                img_ds = DetDataSample()
                img_ds.pred_track_instances = inst
            else:
                img_ds = ts[0]  # type: ignore[index]
            video_samples.append(img_ds)
        vds = TrackDataSample()
        vds.video_data_samples = video_samples
        try:
            vds.set_metainfo({"key_frames_inds": list(range(len(video_samples)))})
        except Exception:
            pass
        return vds
