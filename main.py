import logging
from tkinter import Tk

from detect_crystals import DetectCrystals

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    root = Tk()
    app = DetectCrystals(root)
    root.mainloop()
