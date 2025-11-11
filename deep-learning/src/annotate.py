import cv2


class Annotator:

    def __init__(self):
        self.frame = None  
    
    def set_frame(self, frame):
        self.frame = frame

    def get_frame(self):
        return self.frame

    def annotate_bounding_box(self, bounding_box, conf, color = (255, 255, 255)) -> None:
        """
        Draw a bounding box with label on the frame.
        
        Args:
            frame: Video frame to annotate
            box: list of bounding box coordinates [x1, y1, x2, y2]
            id: Label id corresponding to the kalman filter its associated with.
            conf: Confidence score
            color: BGR color tuple for the box and text
        
        Returns:
            Annotated frame
        """
        x1, y1, x2, y2 = map(int, bounding_box)
        label_text = f"Conf:{conf:.2f}"
        cv2.rectangle(self.frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(self.frame, label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def annotate_tracker(self, px, py, end_x, end_y, pred_x, pred_y, color, history) -> None:
        # Arrow Showing the vector to the next predicted point
        cv2.arrowedLine(self.frame, (int(px), int(py)), (end_x, end_y), color, 2, tipLength=0.3)

        # Circle showing the position of the next predicted point
        cv2.circle(self.frame, (int(pred_x), int(pred_y)), 8, color, 4)

        # For debugging, show the last 5 positions in history.
        for x, y in history:
            cv2.circle(self.frame, (int(x), int(y)), 8, color, -1)
        

    def annotate_trackers(self, trackers, hits) -> None:
        for t in trackers:
            px, py, vx, vy, _, _ = t.get_state()
            pred_x , pred_y = t.predict()

            if t.id in hits:
                self.annotate_hit(px, py)     
            
            # Annotate all things needed for tracker
            self.annotate_tracker(
                px=px,
                py=py, 
                end_x=int(px + vx),
                end_y=int(py + vy),
                pred_x=pred_x,
                pred_y=pred_y,
                color= t.color, 
                history=list(t.history)[-5:]
            )

    def annotate_bounding_boxes(self, boxes, confs) -> None:
        # Annotate Bounding Boxes
        for box, conf in zip(boxes, confs):
            self.annotate_bounding_box(bounding_box=box, conf=conf)

    def annotate_hit(self, px, py) -> None:
        cv2.putText(self.frame, "HIT!", (int(px) + 30, int(py) + 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        
    
    def annotate_time(self, time_s: float) -> None:
        cv2.putText(self.frame,  f"{time_s}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
