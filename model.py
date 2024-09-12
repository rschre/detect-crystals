import logging
import os
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import FastSAM

logger = logging.getLogger(__name__)


def get_model(checkpoint_path: str):
    """Returns a FastSAM model."""
    return FastSAM(checkpoint_path or "./model/FastSAM-x.pt")


def get_filtered_boxes(result, min_area=None, max_area=None):
    """Returns a DataFrame of the detected boxes."""

    min_area = min_area or 10000
    max_area = max_area or 250000

    boxes = [
        {
            "xmin": box[0].item(),
            "ymin": box[1].item(),
            "xmax": box[2].item(),
            "ymax": box[3].item(),
        }
        for box in result[0].boxes.xyxy
    ]
    boxes = [
        {**box, "area": (box["xmax"] - box["xmin"]) * (box["ymax"] - box["ymin"])}
        for box in boxes
    ]

    df_boxes = pd.DataFrame(columns=["xmin", "ymin", "xmax", "ymax", "area"])

    for i, box in enumerate(boxes):
        if min_area < box["area"] < max_area:
            df_boxes.loc[i] = pd.Series(box)

    return df_boxes


def process_dataset(
    model: FastSAM,
    files: List,
    roi: List,
    confidence_threshold: float,
    iou_threshold: float,
    show_detection: bool = False,
    save_detection: bool = False,
    output_dir: str = "./output",
):
    """Processes a dataset of images using a FastSAM model."""
    df_res = pd.DataFrame(columns=["file", "object_counter", "object_area_sum"])

    for i, f in enumerate(files):
        object_counter = 0
        object_area_sum = 0

        logger.info(f"Processing {f}")

        img = cv2.imread(f)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        logger.debug(f"Image shape: {img.shape}")

        img = img[
            int(roi[1]) : int(roi[1] + roi[3]), int(roi[0]) : int(roi[0] + roi[2])
        ]
        logger.debug(f"Cropped shape: {img.shape}")

        res = model.predict(
            img,
            device="cpu",
            retina_masks=True,
            conf=confidence_threshold,
        )

        if len(res[0].boxes) == 0:
            logger.info("No objects detected")
            continue

        for idx, row in df_boxes.iterrows():
            if 10000 < row["bbox_area"] < 250000:
                object_counter += 1
                object_area_sum += row["bbox_area"]
                if not show_detection or save_detection:
                    continue

                cv2.rectangle(
                    img,
                    (int(row["xmin"]), int(row["ymin"])),
                    (int(row["xmax"]), int(row["ymax"])),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    img,
                    f"{row['bbox_area']:.0f}",
                    (int(row["xmin"]), int(row["ymin"])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

                if save_detection:
                    cv2.imwrite(f"{output_dir}/{os.path.basename(f)}", img)

        df_res.loc[i] = pd.Series(
            {
                "file": os.path.basename(f),
                "object_counter": object_counter,
                "object_area_sum": object_area_sum,
            }
        )

        logger.info(
            f"Detected {object_counter} objects with a total area of {object_area_sum:.0f}"
        )

    return df_res
