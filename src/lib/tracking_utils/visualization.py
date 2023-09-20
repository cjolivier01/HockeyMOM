import copy
import numpy as np
import math
import cv2
from typing import List, Tuple

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def tlwhs_to_tlbrs(tlwhs):
    tlbrs = np.copy(tlwhs)
    if len(tlbrs) == 0:
        return tlbrs
    tlbrs[:, 2] += tlwhs[:, 0]
    tlbrs[:, 3] += tlwhs[:, 1]
    return tlbrs


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


def resize_image(image, max_size=800):
    if max(image.shape[:2]) > max_size:
        scale = float(max_size) / max(image.shape[:2])
        image = cv2.resize(image, None, fx=scale, fy=scale)
    return image


def plot_rectangle(
    img, box: List[int], color: Tuple[int, int, int], thickness: int, label: str = None
):
    intbox = [int(i) for i in box]
    cv2.rectangle(
        img,
        intbox[0:2],
        intbox[2:4],
        color=color,
        thickness=thickness,
    )
    if label:
        text_scale = 1
        text_thickness = 2
        cv2.putText(
            img,
            label,
            (intbox[0], intbox[1] + 30),
            cv2.FONT_HERSHEY_PLAIN,
            text_scale,
            (0, 0, 128),
            thickness=text_thickness,
        )


def _to_int(vals):
    return [int(i) for i in vals]


def plot_line(img, src_point, dest_point, color: Tuple[int, int, int], thickness: int):
    cv2.line(img, _to_int(src_point), _to_int(dest_point), color=color, thickness=thickness)


def plot_point(img, point, color: Tuple[int, int, int], thickness: int):
    x = int(point[0] + 0.5 * thickness)
    y = int(point[1] + 0.5 * thickness)
    cv2.circle(img, [x, y], radius=int((thickness + 1)//2), color=color, thickness=thickness)

last_frame_id = -1


def plot_frame_id_and_speeds(im, frame_id, vel_x, vel_y, accel_x, accel_y):
    text_scale = max(2, im.shape[1] / 1600.0)
    text_thickness = 2
    line_thickness = 1

    y_delta = int(15 * text_scale)
    text_y_offset = y_delta

    cv2.putText(
        im,
        "frame: %d"
        % (frame_id),
        (0, text_y_offset),
        cv2.FONT_HERSHEY_PLAIN,
        text_scale,
        (0, 0, 255),
        thickness=2,
    )

    text_y_offset += y_delta
    cv2.putText(
        im,
        "vel_x: %.2f"
        % (vel_x),
        (0, text_y_offset),
        cv2.FONT_HERSHEY_PLAIN,
        text_scale,
        (0, 0, 255),
        thickness=2,
    )

    text_y_offset += y_delta
    cv2.putText(
        im,
        "vel_y: %.2f"
        % (vel_y),
        (0, text_y_offset),
        cv2.FONT_HERSHEY_PLAIN,
        text_scale,
        (0, 0, 255),
        thickness=2,
    )

    text_y_offset += y_delta
    cv2.putText(
        im,
        "accel_x: %.2f"
        % (accel_x),
        (0, text_y_offset),
        cv2.FONT_HERSHEY_PLAIN,
        text_scale,
        (0, 0, 255),
        thickness=2,
    )

    text_y_offset += y_delta
    cv2.putText(
        im,
        "accel_y: %.2f"
        % (accel_y),
        (0, text_y_offset),
        cv2.FONT_HERSHEY_PLAIN,
        text_scale,
        (0, 0, 255),
        thickness=2,
    )


def plot_tracking(
    image,
    tlwhs,
    obj_ids,
    frame_id,
    scores=None,
    fps=0.0,
    ids2=None,
    speeds: List[float] = None,
    box_color: Tuple[int, int, int] = None,
    show_frame_heading: bool = False,
    line_thickness: int = 1,
):
    global last_frame_id
    # don't call this more than once per frame
    assert frame_id > last_frame_id
    last_frame_id = frame_id
    assert len(tlwhs) == len(obj_ids)
    if speeds:
        assert len(speeds) == len(obj_ids)
    # TODO: is this an unnecessary copy?
    im = np.ascontiguousarray(image)
    # im = image
    # im_h, im_w = im.shape[:2]
    # im_h, im_w = im.shape[:2]

    # top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    text_scale = max(2, image.shape[1] / 1600.0)
    text_thickness = 2
    text_offset = int(8 * text_scale)

    if show_frame_heading:
        cv2.putText(
            im,
            "frame: %d fps: %.2f num: %d" % (frame_id, fps, len(tlwhs)),
            (0, int(15 * text_scale)),
            cv2.FONT_HERSHEY_PLAIN,
            text_scale,
            (0, 0, 255),
            thickness=2,
        )

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        id_text = "{}".format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ", {}".format(int(ids2[i]))
        color = box_color if box_color is not None else get_color(abs(obj_id))
        cv2.rectangle(
            im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness
        )
        cv2.putText(
            im,
            id_text,
            (intbox[0], intbox[1] + text_offset),
            cv2.FONT_HERSHEY_PLAIN,
            text_scale,
            (0, 0, 255),
            thickness=text_thickness,
        )
        if speeds:
            speed = speeds[i]
            if not np.isnan(speed):
                if speed < 10:
                    speed_text = "{:.1f}".format(speeds[i])
                else:
                    speed_text = f"{int(speed + 0.5)}"
            else:
                speed_text = "NaN"
            pos_x = intbox[2] - text_offset
            pos_y = intbox[3]
            cv2.putText(
                im,
                speed_text,
                (pos_x, pos_y),
                cv2.FONT_HERSHEY_PLAIN,
                text_scale,
                (150, 0, 150),
                thickness=text_thickness,
            )
    return im


# def plot_camera(image, obj_ids: List[int], hockey_mom: HockeyMOM):
#     original_bounding_box = hockey_mom.get_current_bounding_box(obj_ids=obj_ids)
#     print(original_bounding_box)
#     cv2.rectangle(
#         image,
#         original_bounding_box[0:2],
#         original_bounding_box[2:4],
#         color=(0, 0, 0),
#         thickness=6,
#     )
#     x1 = int(original_bounding_box[0])
#     y1 = int(original_bounding_box[1])
#     w = int(original_bounding_box[2])
#     h = int(original_bounding_box[3])
#     cv2.circle(
#         image, (int(x1 + 0.5 * w), int(y1 + h)), 2, color=(255, 255, 0), thickness=20
#     )
#     return image


def plot_trajectory(image, tlwhs, track_ids):
    assert len(tlwhs) == len(track_ids)
    for one_tlwhs, track_id in zip(tlwhs, track_ids):
        color = get_color(int(track_id))
        for tlwh in one_tlwhs:
            x1 = int(tlwh[0])
            y1 = int(tlwh[1])
            w = int(tlwh[2])
            h = int(tlwh[3])
            cv2.circle(image, (int(x1 + 0.5 * w), int(y1 + h)), 2, color, thickness=2)

    return image


def plot_detections(image, tlbrs, scores=None, color=(255, 0, 0), ids=None):
    im = np.copy(image)
    text_scale = max(1, image.shape[1] / 800.0)
    thickness = 2 if text_scale > 1.3 else 1
    for i, det in enumerate(tlbrs):
        x1, y1, x2, y2 = np.asarray(det[:4], dtype=np.int)
        if len(det) >= 7:
            label = "det" if det[5] > 0 else "trk"
            if ids is not None:
                text = "{}# {:.2f}: {:d}".format(label, det[6], ids[i])
                cv2.putText(
                    im,
                    text,
                    (x1, y1 + 30),
                    cv2.FONT_HERSHEY_PLAIN,
                    text_scale,
                    (0, 255, 255),
                    thickness=thickness,
                )
            else:
                text = "{}# {:.2f}".format(label, det[6])

        if scores is not None:
            text = "{:.2f}".format(scores[i])
            cv2.putText(
                im,
                text,
                (x1, y1 + 30),
                cv2.FONT_HERSHEY_PLAIN,
                text_scale,
                (0, 255, 255),
                thickness=thickness,
            )

        cv2.rectangle(im, (x1, y1), (x2, y2), color, 2)

    return im


# def plot_kmeans_intertias(hockey_mom: HockeyMOM):
#     inertias = []
#     object_count = len(hockey_mom.online_image_center_points)
#     for i in range(1, object_count):
#         kmeans = KMeans(n_clusters=i, n_init="auto")
#         kmeans.fit(hockey_mom.online_image_center_points)
#         inertias.append(kmeans.inertia_)

#     plt.plot(range(1, object_count), inertias, marker="o")
#     plt.title("Elbow method")
#     plt.xlabel("Number of clusters")
#     plt.ylabel("Inertia")
#     plt.show()


# def plot_kmeans_scatter(hockey_mom: HockeyMOM, n_clusters: int = 3):
#     kmeans = KMeans(n_clusters=n_clusters)
#     kmeans.fit(hockey_mom.online_image_center_points)
#     for x, y in hockey_mom.online_image_center_points:
#         plt.scatter(x, y, c=kmeans.labels_)
#     plt.show()
