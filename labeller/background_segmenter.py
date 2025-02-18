from queue import SimpleQueue

import numpy as np
from PySide6.QtGui import QStatusTipEvent
from PySide6.QtWidgets import QApplication

from database import DatabaseFrame, Record, active_db
from drawing import draw_clicks, update_frame_image
from main_window import MainWindow
from sam2_processor import Sam2Processor


class BackgroundSegmenter:
    def __init__(self, window: MainWindow, work_queue: SimpleQueue) -> None:
        self.should_stop = False
        self.window = window

        self.run_with_sam = True
        if self.run_with_sam:
            self.sam2_ = Sam2Processor()
        else:
            self.sam2_ = None

        self.work_queue_ = work_queue

    def run(self) -> None:
        while not self.should_stop:
            if QApplication.activeWindow() is not None:
                if self.work_queue_.empty():
                    QApplication.sendEvent(
                        QApplication.activeWindow(), QStatusTipEvent("sam2:Ready")
                    )

            try:
                frame_index: int = self.work_queue_.get(block=True, timeout=0.5)
            except:
                continue

            frame = active_db().frames.get(frame_index)

            # Check that frame was not deleted
            if frame is not None:
                if QApplication.activeWindow() is not None:
                    QApplication.sendEvent(
                        QApplication.activeWindow(),
                        QStatusTipEvent(
                            f"sam2:Segmenting {self.work_queue_.qsize() + 1} frames..."
                        ),
                    )

                # Do a slow segmentation
                for record in frame.records.values():
                    if record.segmentation is None:
                        # Segment!
                        self.segment_record(frame, record)

                # Combine segmentations into a single image
                update_frame_image(frame)

            # Trigger UI update
            self.window.update_ui(frame_index)

    def segment_record(self, frame: DatabaseFrame, record: Record) -> None:
        if self.sam2_:
            mask = self.sam2_.process_click(
                frame.original_image, record.positive_points, record.negative_points
            )
            mask = mask.astype(np.uint8)
        else:
            mask = np.full(
                (frame.original_image.shape[0], frame.original_image.shape[1]), 0
            )
        record.segmentation = mask
