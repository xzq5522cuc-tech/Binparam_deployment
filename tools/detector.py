"""FP32 YOLOv8 ONNX person detection; no torch/Ultralytics dependency."""
import time
import cv2
import numpy as np


class PersonDetector:
    def __init__(self, path, size=320, confidence=.20, redetect_seconds=0):
        self.net = cv2.dnn.readNetFromONNX(str(path))
        self.size = size
        self.confidence = confidence
        self.previous = None
        self.redetect_seconds = redetect_seconds
        self.last_detection = 0
        self.gray = None
        self.points = None
        self.mode = 'detect'

    def detect(self, frame):
        begin = time.monotonic()
        tracking_ratio = min(1., 320./frame.shape[1])
        small = cv2.resize(frame, (round(frame.shape[1]*tracking_ratio),
                                   round(frame.shape[0]*tracking_ratio)))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if (self.previous is not None and self.points is not None and self.gray is not None
                and gray.shape == self.gray.shape
                and begin-self.last_detection < self.redetect_seconds):
            nxt, status, error = cv2.calcOpticalFlowPyrLK(self.gray, gray, self.points, None)
            if nxt is not None:
                back, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, self.gray, nxt, None)
                if back is not None:
                    good = ((status.ravel() == 1) & (back_status.ravel() == 1)
                            & (np.linalg.norm(back-self.points,axis=2).ravel() < 1.5)
                            & (error.ravel() < 25))
                    if good.sum() >= 8:
                        delta = np.median((nxt-self.points)[good,0,:],axis=0)/tracking_ratio
                        x,y,w,h = self.previous
                        nx,ny = x+float(delta[0]),y+float(delta[1])
                        if (np.linalg.norm(delta) < .25*max(w,h) and nx >= 0 and ny >= 0
                                and nx+w <= frame.shape[1] and ny+h <= frame.shape[0]):
                            self.previous = [nx,ny,w,h]
                            self.gray, self.points = gray, nxt[good].reshape(-1,1,2)
                            self.mode = 'optical_flow'
                            return self.previous, (time.monotonic()-begin)*1000
        box, _ = self._detect(frame)
        self.last_detection = time.monotonic()
        self.mode = 'detect'
        self.gray, self.points = gray, None
        if box is not None and self.redetect_seconds > 0:
            x,y,w,h = box
            mask = np.zeros_like(gray)
            # Prefer interior points; still only a short-lived motion tracker, not identity tracking.
            mask[int((y+.1*h)*tracking_ratio):int((y+.9*h)*tracking_ratio),
                 int((x+.1*w)*tracking_ratio):int((x+.9*w)*tracking_ratio)] = 255
            self.points = cv2.goodFeaturesToTrack(gray,60,.02,7,mask=mask)
        return box, (time.monotonic()-begin)*1000

    def _detect(self, frame):
        begin = time.monotonic()
        h, w = frame.shape[:2]
        ratio = min(self.size / w, self.size / h)
        nw, nh = round(w * ratio), round(h * ratio)
        left, top = (self.size - nw) // 2, (self.size - nh) // 2
        image = np.full((self.size, self.size, 3), 114, np.uint8)
        image[top:top+nh, left:left+nw] = cv2.resize(frame, (nw, nh))
        self.net.setInput(cv2.dnn.blobFromImage(image, 1/255., swapRB=True))
        raw = self.net.forward()
        if raw.ndim != 3 or raw.shape[1] != 84:
            raise ValueError(f'Expected YOLOv8 COCO output [1,84,N], got {raw.shape}')
        rows = raw[0].T
        # Match person-only inference: class 0 confidence, not objectness.
        rows = rows[rows[:, 4] >= self.confidence]
        boxes, scores = [], []
        for row in rows:
            cx, cy, bw, bh = row[:4]
            x1 = float(np.clip((cx-bw/2-left)/ratio, 0, w))
            y1 = float(np.clip((cy-bh/2-top)/ratio, 0, h))
            x2 = float(np.clip((cx+bw/2-left)/ratio, 0, w))
            y2 = float(np.clip((cy+bh/2-top)/ratio, 0, h))
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2-x1, y2-y1])
                scores.append(float(row[4]))
        keep = np.asarray(cv2.dnn.NMSBoxes(boxes, scores, self.confidence, .45)).reshape(-1)
        if not keep.size:
            self.previous = None
            return None, (time.monotonic()-begin)*1000
        def iou(a, b):
            area = max(0, min(a[0]+a[2], b[0]+b[2])-max(a[0], b[0])) * max(0, min(a[1]+a[3], b[1]+b[3])-max(a[1], b[1]))
            return area / max(a[2]*a[3]+b[2]*b[3]-area, 1e-9)
        selected = max(keep, key=lambda i: boxes[i][2]*boxes[i][3]*scores[i])
        if self.previous is not None:
            matched = max(keep, key=lambda i: iou(boxes[i], self.previous))
            if iou(boxes[matched], self.previous) >= .3:
                selected = matched
        self.previous = boxes[selected]
        return self.previous, (time.monotonic()-begin)*1000
