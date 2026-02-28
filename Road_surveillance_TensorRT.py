import cv2
import time
import numpy as np
from collections import deque
from dataclasses import dataclass
from sort import Sort
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


# Configuration
@dataclass
class Config:
    VIDEO_PATH: str = "/Users/29924/Desktop/HKU_FYP_ALL/testhku.mp4"
    ENGINE_PATH: str = "C:/Users/29924/Desktop/yolov5/yolov5-7.0/runs/train/exp5/weights/yolov5s_final.engine"

    INPUT_SIZE: int = 640
    YOLO_THRESHOLD: float = 0.6

    # DVS
    DVS_THRESHOLD: float = 0.8
    ACCUMULATION_TIME: float = 0.001

config = Config()


# TensorRT YOLOv5 Detector
class TRT_YOLOv5:

    def __init__(self, engine_path, conf_thres=0.6, iou_thres=0.45):

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        self.class_map = {
            0: "Car",
            1: "Rider",
            2: "Pedestrian"
        }

        TRT_LOGGER = trt.Logger(trt.Logger.INFO)

        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.bindings = []

        for i in range(self.engine.num_bindings):
            shape = self.engine.get_binding_shape(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            size = int(np.prod(shape))

            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))

            if self.engine.binding_is_input(i):
                self.input_host = host_mem
                self.input_device = device_mem
                self.input_shape = shape
            else:
                self.output_host = host_mem
                self.output_device = device_mem
                self.output_shape = shape

    def letterbox(self, img):
        h, w = img.shape[:2]
        input_h, input_w = self.input_shape[2], self.input_shape[3]

        r = min(input_w / w, input_h / h)
        nw, nh = int(w * r), int(h * r)

        resized = cv2.resize(img, (nw, nh))
        canvas = np.full((input_h, input_w, 3), 114, dtype=np.uint8)

        dw = (input_w - nw) // 2
        dh = (input_h - nh) // 2
        canvas[dh:dh+nh, dw:dw+nw] = resized

        return canvas, r, dw, dh

    def infer(self, image):

        img, r, dw, dh = self.letterbox(image)

        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        np.copyto(self.input_host, img.ravel())

        cuda.memcpy_htod_async(self.input_device, self.input_host, self.stream)
        self.context.execute_async_v2(self.bindings, self.stream.handle)
        cuda.memcpy_dtoh_async(self.output_host, self.output_device, self.stream)
        self.stream.synchronize()

        preds = self.output_host.reshape(self.output_shape)

        boxes, scores, classes = [], [], []
        preds = preds.reshape(-1, preds.shape[-1])

        for det in preds:

            conf = float(det[4])
            if conf < self.conf_thres:
                continue

            cls = int(np.argmax(det[5:]))
            score = conf * float(det[5 + cls])
            if score < self.conf_thres:
                continue

            x, y, w, h = det[:4]

            x = (x - dw) / r
            y = (y - dh) / r
            w /= r
            h /= r

            x1 = int(x - w/2)
            y1 = int(y - h/2)
            x2 = int(x + w/2)
            y2 = int(y + h/2)

            # TensorFlow格式 (y1,x1,y2,x2)
            boxes.append([y1, x1, y2, x2])
            scores.append(score)
            classes.append(cls)

        indices = cv2.dnn.NMSBoxes(boxes, scores,
                                   self.conf_thres,
                                   self.iou_thres)

        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                detections.append((boxes[i], scores[i], classes[i]))

        return detections

# =========================
# DVS Encoder
# =========================
class DVSEncoder:

    def __init__(self, first_frame, config):
        gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        self.prev_gray = np.log1p(gray + 1)
        self.config = config

    def encode(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray = np.log1p(gray + 1)

        delta = gray - self.prev_gray
        self.prev_gray = gray

        pos = delta > self.config.DVS_THRESHOLD
        neg = delta < -self.config.DVS_THRESHOLD

        evt = np.zeros_like(frame)
        evt[pos] = (0,255,0)
        evt[neg] = (0,0,255)

        return evt


# SORT Tracker
class SortTracker:

    def __init__(self,
                 max_age=20,
                 min_hits=2,
                 iou_threshold=0.3,
                 trail_length=30):

        self.tracker = Sort(
            max_age=max_age,
            min_hits=min_hits,
            iou_threshold=iou_threshold
        )

        self.track_history = {}
        self.trail_length = trail_length

    def update(self, detections):
        """
        :sort detections format: [(box, conf, cls), ...]
                            : box = [x1, y1, x2, y2]
        """

        dets_for_sort = []

        for box, conf, _ in detections:
            y1, x1, y2, x2 = box
            dets_for_sort.append([x1, y1, x2, y2, conf])

        if len(dets_for_sort) > 0:
            dets_for_sort = np.array(dets_for_sort)
        else:
            dets_for_sort = np.empty((0, 5))

        tracks = self.tracker.update(dets_for_sort)

        return tracks

    def draw(self, frame, tracks):
        """
        Draw tracking results on frame
        """

        for track in tracks:
            x1, y1, x2, y2, track_id = track.astype(int)

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            if track_id not in self.track_history:
                self.track_history[track_id] = []

            self.track_history[track_id].append((cx, cy))

            if len(self.track_history[track_id]) > self.trail_length:
                self.track_history[track_id].pop(0)

            # Draw trajectory
            for i in range(1, len(self.track_history[track_id])):
                cv2.line(frame,
                         self.track_history[track_id][i - 1],
                         self.track_history[track_id][i],
                         (0, 255, 255),
                         2)

            # Draw ID
            text_y = max(20, y1 - 25)

            cv2.putText(frame,
                        f"ID {track_id}",
                        (x1, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2)


# Main
def main():

    cap = cv2.VideoCapture(config.VIDEO_PATH)
    ret, first_frame = cap.read()

    dvs = DVSEncoder(first_frame, config)
    detector = TRT_YOLOv5(config.ENGINE_PATH)
    tracker = SortTracker()

    fps_buffer = deque(maxlen=30)
    frame_idx = 0

    while True:

        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        dvs_frame = dvs.encode(frame)
        detections = detector.infer(dvs_frame)

        tracks = tracker.update(detections)
        tracker.draw(dvs_frame, tracks)

        # ===== Draw detection =====
        for idx, (box, conf, cls) in enumerate(detections, 1):
            y1, x1, y2, x2 = box
            label = detector.class_map.get(cls, "unknown")

            cv2.rectangle(dvs_frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(dvs_frame,
                        f"{idx}:{label} {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0,255,0),
                        2)

        cv2.imshow("DVS_TRTYOLO_SORT", dvs_frame)

        fps_buffer.append(time.time() - t0)
        if frame_idx % 10 == 0 and len(fps_buffer) > 1:
            fps = 1.0 / (sum(fps_buffer) / len(fps_buffer))
            print(f"[{frame_idx}] ≈ {fps:.2f} FPS")

        frame_idx += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
