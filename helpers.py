import glob
import logging
import os

import cv2
import numpy as np
import pandas as pd
import toml
from numpy.typing import ArrayLike

logger = logging.getLogger(__name__)


def load_config(path: str):
    """Loads a TOML config file and returns a dict of the values.
    If the capture directory is not defined, prompts the user to select one.

    Args:
        path (str): Path to the TOML config file.

    Returns:
        dict: Dictionary of the config values."""

    try:
        config = toml.load(path)

    except toml.TomlDecodeError:
        logger.error(
            f"Config file is not a valid TOML file. Verify that {path} is valid."
        )
        exit(1)
    except FileNotFoundError:
        logger.error(f"Config file not found. Verify that {path} exists.")
        exit(1)

    if (
        config.get("storage").get("data_folder") is None
        or config.get("storage").get("data_folder") == ""
    ):
        logger.debug("No capture path defined in config file. Defaulting to Desktop")
        # get current users desktop path
        data_loc = os.path.join(os.path.expanduser("~"), "Desktop")
        if not os.path.exists(data_loc):
            data_loc = glob.glob(os.path.join(os.path.expanduser("~"), "*", "Desktop"))[
                0
            ]

        if data_loc == "":
            print("No folder selected. Exiting.")
            exit(0)
        else:
            config["storage"]["data_folder"] = data_loc

    return config


def select_roi(img: np.ndarray, screen_width: int = 1920, screen_height: int = 1080):
    """Selects a region of interest from an image.
    Args:
        img (np.ndarray): Image to select the ROI from.

    Returns:
        tuple: Tuple of the top left and bottom right coordinates of the ROI.
    """
    window = cv2.namedWindow("Select Range of Interest", cv2.WINDOW_NORMAL)
    print(img.shape)
    if img.shape[1] // 2 > screen_width or img.shape[0] // 2 > screen_height:
        cv2.resizeWindow(
            "Select Range of Interest", img.shape[1] // 4, img.shape[0] // 4
        )
    elif img.shape[1] > screen_width or img.shape[0] > screen_height:
        cv2.resizeWindow(
            "Select Range of Interest", img.shape[1] // 2, img.shape[0] // 2
        )
    else:
        cv2.resizeWindow("Select Range of Interest", img.shape[1], img.shape[0])

    text = [
        "Drag to select the region of interest.",
        "Press Enter to confirm.",
        "Press Esc to cancel.",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)
    font_thickness = 1
    org = (150, 150)

    for i, line in enumerate(text):
        cv2.putText(
            img,
            line,
            (org[0], org[1] + i * 50),
            font,
            font_scale,
            font_color,
            font_thickness,
        )

    roi = cv2.selectROI("Select Range of Interest", img, fromCenter=False)
    cv2.destroyWindow("Select Range of Interest")
    return roi


def draw_bboxes(img: ArrayLike, boxes: pd.DataFrame) -> ArrayLike:
    """Draws bounding boxes on an image.

    Args:
        img (ArrayLike): Image to draw the bounding boxes on.
        boxes (List): List of bounding boxes to draw.

    Returns:
        ArrayLike: Image with bounding boxes drawn.
    """
    for i, row in boxes.iterrows():
        cv2.rectangle(
            img,
            (int(row["xmin"]), int(row["ymin"])),
            (int(row["xmax"]), int(row["ymax"])),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            img,
            f"{row['area']:.0f}",
            (int(row["xmin"]), int(row["ymin"])),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

    return img
