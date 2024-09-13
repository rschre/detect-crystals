import logging
import os
from glob import glob
from tkinter import *
from tkinter import filedialog, messagebox, ttk

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from numpy.typing import ArrayLike
from scipy.stats import norm

from helpers import draw_bboxes, load_config, select_roi
from model import get_filtered_boxes, get_model

logger = logging.getLogger(__name__)


class DetectCrystals:
    def __init__(self, root, config_path=None):
        self.root = root
        self.root.title("Crystal Detection")
        self.root.geometry("550x110")
        self.config = load_config(config_path or "config.toml")
        self._set_vars()
        self._create_ui()
        self.model = None

    def _create_ui(self):
        frame = ttk.Frame(self.root, padding="3 3 12 12")
        frame.grid(column=0, row=0, sticky=(N, W, E, S))
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        ttk.Label(frame, text="Dataset Directory", width=20).grid(
            column=1, row=1, sticky=W
        )
        conf_thresh_entry = ttk.Entry(frame, textvariable=self.data_folder, width=50)
        conf_thresh_entry.grid(column=2, row=1, sticky=(W, E))

        data_folder_button = ttk.Button(frame, text="...", command=self._select_folder)
        data_folder_button.grid(column=3, row=1, sticky=W)

        ttk.Label(frame, text="Confidence Threshold", width=20).grid(
            column=1, row=2, sticky=W
        )
        conf_thresh_entry = ttk.Entry(
            frame, textvariable=self.model_confidence_threshold, width=50
        )
        conf_thresh_entry.grid(column=2, row=2, sticky=(W, E))

        ttk.Label(frame, text="IoU Threshold", width=20).grid(column=1, row=3, sticky=W)
        iou_entry = ttk.Entry(frame, textvariable=self.model_iou_threshold, width=50)
        iou_entry.grid(column=2, row=3, sticky=(W, E))

        show_detection_check = ttk.Checkbutton(
            frame, text="Show Masks", variable=self.show_detection
        )
        show_detection_check.grid(column=1, row=4, sticky=W)

        show_detection_check = ttk.Checkbutton(
            frame, text="Save Masks", variable=self.save_detection
        )
        show_detection_check.grid(column=2, row=4, sticky=W)

        ttk.Button(frame, text="Detect Crystals", command=self._detect_crystals).grid(
            column=3, row=4, sticky=W
        )

        # TODO: Enable this again
        # self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _set_vars(self):
        self.data_folder = StringVar()
        self.output_folder = StringVar()
        self.model_confidence_threshold = DoubleVar()
        self.model_iou_threshold = DoubleVar()
        self.show_detection = BooleanVar()
        self.save_detection = BooleanVar()
        self.progress = IntVar()
        self.progress_text = StringVar()

        self.data_folder.set(self.config["storage"]["data_folder"])
        self.output_folder.set(self.config["storage"]["output_folder"])
        self.model_confidence_threshold.set(
            self.config["model"]["confidence_threshold"]
        )
        self.model_iou_threshold.set(self.config["model"]["iou_threshold"])
        self.show_detection.set(self.config["model"]["show_detection"])
        self.save_detection.set(self.config["model"]["save_detection"])

    def _create_loading_window(self, maximum):
        self.loading_window = Toplevel(self.root)
        self.loading_window.title("Detecting Crystals")
        self.loading_window.geometry("300x100")
        self.loading_window.grab_set()

        maximum = maximum or 100

        self.progress.set(0)
        self.progress_text.set("Detecting Crystals...")

        self.loading_label = ttk.Label(
            self.loading_window, textvariable=self.progress_text
        ).pack()
        self.progress_bar = ttk.Progressbar(
            self.loading_window,
            mode="determinate",
            variable=self.progress,
            maximum=maximum,
        ).pack()

        self.loading_window.attributes("-topmost", True)
        self.loading_window.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.root.update()

    def _update_progress(self, step):
        current_progress = self.progress.get()
        self.progress.set(current_progress + step)

    def _detect_crystals(self):
        if not self.model:
            self.model = get_model(self.config["model"]["checkpoint_path"])

        data_folder = self.data_folder.get()
        files = glob(f"{data_folder}/*.bmp") + glob(f"{data_folder}/*.png")

        if len(files) == 0:
            messagebox.showerror(
                "Error", "No images found in the selected folder. Exiting."
            )
            return

        first_img = cv2.imread(files[0])
        first_img = cv2.cvtColor(first_img, cv2.COLOR_BGR2RGB)

        roi = select_roi(
            first_img,
            screen_width=self.root.winfo_screenwidth(),
            screen_height=self.root.winfo_screenheight(),
        )

        if roi is None:
            messagebox.showerror("Error", "No region of interest selected. Exiting.")
            return

        self._create_loading_window(len(files))

        df_res = pd.DataFrame(columns=["file", "object_counter", "object_area_sum"])

        for i, file in enumerate(files):
            self.progress_text.set(f"Processing file {i+1}/{len(files)}")
            self.root.update()
            logger.info(f"Processing {file}")

            img = cv2.imread(file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            logger.debug(f"Image shape: {img.shape}")

            img = img[
                int(roi[1]) : int(roi[1] + roi[3]), int(roi[0]) : int(roi[0] + roi[2])
            ]
            logger.debug(f"Cropped shape: {img.shape}")

            res = self.model.predict(
                img,
                device="cpu",
                retina_masks=True,
                conf=self.model_confidence_threshold.get(),
                iou=self.model_iou_threshold.get(),
            )

            if len(res[0].boxes) == 0:
                logger.info("No objects detected")
                continue

            res_boxes = get_filtered_boxes(res)
            logger.debug(f"Filtered boxes: {res_boxes}")
            if self.show_detection.get():
                img_boxes = draw_bboxes(img, res_boxes)
                self._show_image(
                    img_boxes,
                    title=f"{os.path.basename(file)}, Objects: {len(res_boxes)}",
                )

            df_res.loc[i] = pd.Series(
                {
                    "file": os.path.basename(file),
                    "object_counter": len(res_boxes),
                    "object_area_sum": res_boxes["area"].sum(),
                }
            )
            df_res = df_res.astype({"object_counter": int, "object_area_sum": int})
            self._update_progress(1)

        self.loading_window.destroy()

        print(df_res)
        df_res.hist(
            column="object_counter",
            bins=df_res["object_counter"].max() + 1,
            zorder=2,
            rwidth=0.9,
        )
        plt.xlabel("Number of objects")
        plt.ylabel("Number of frames")
        plt.title("Number of objects per frame")
        plt.grid(axis="y", zorder=1)

        mean = df_res["object_counter"].mean()
        std = df_res["object_counter"].std()
        x = np.linspace(0, df_res["object_counter"].max(), 100)
        y = len(df_res) * norm.pdf(x, mean, std)
        plt.plot(x, y, "r--", linewidth=2)
        plt.show()

        df_res.to_csv(f"{self.data_folder.get()}/results.csv", index=False)

    def _show_image(
        self,
        img: ArrayLike | Image,
        title: str = None,
        x_offset: int = 0,
        y_offset: int = 0,
    ):
        """Creates a new window to display an image. Normally it is placed in the center of the screen.
        The image is displayed using matplotlib and the window is resizable.
        The window will close when the user closes it.

        Args:
            img (ArrayLike | Image): The image to display. Needs to be in a matplotlib-compatible format.
            title (str): The title of the window. Defaults to "Image".
            x_offset (int): The x offset to position the window.
            y_offset (int): The y offset to position the window.
        """
        img_view = Toplevel(self.root)
        title = title or "Image"
        img_view.title(title)
        h, w = 750, 600
        ws = self.root.winfo_screenwidth()
        hs = self.root.winfo_screenheight()

        x = (ws / 2) - (w / 2) + x_offset
        y = (hs / 2) - (h / 2) - 50 + y_offset
        img_view.geometry(f"{w}x{h}+{int(x)}+{int(y)}")

        fig = Figure(figsize=(5, 4), dpi=100)
        a = fig.add_subplot(111)
        a.title.set_text(title or "Image")
        a.imshow(img)
        canvas = FigureCanvasTkAgg(fig, master=img_view)
        canvas.draw()
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
        toolbar = NavigationToolbar2Tk(canvas, img_view)
        toolbar.update()
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

    def _select_folder(self):
        """Opens a file dialog to select the folder to save the capture to.
        Sets the data_loc variable to the selected folder. Is called when the '...' button is clicked.
        """
        folder = filedialog.askdirectory()
        self.data_folder.set(folder)

    def _on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.root.destroy()
            cv2.destroyAllWindows()
            exit(0)


if __name__ == "__main__":
    root = Tk()
    app = DetectCrystals(root)
    root.mainloop()
