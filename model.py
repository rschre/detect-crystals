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
