import cv2
import numpy as np
import onnxruntime as ort
import os

# 模型與類別設定
onnx_model_path = "C:/Users/user/VS code/yolov8s.onnx"
class_names = ["person", "car", "cup", "book", "cell phone", "bottle"]  # 根據你的模型調整

# 推論參數
input_size = 640
conf_threshold = 0.25
nms_threshold = 0.5

# 建立推論 session
session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

# 定義 letterbox 函式 (保持不變)
def letterbox(img, new_size=640, color=(114, 114, 114)):
    h, w = img.shape[:2]
    scale = new_size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dh, dw = new_size - nh, new_size - nw
    top, bottom = dh // 2, dh - dh // 2
    left, right = dw // 2, dw - dw // 2
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=color)
    return img_padded, scale, left, top

# 開啟攝影機
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_padded, scale, pad_x, pad_y = letterbox(frame, input_size)
    img_input = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_input = np.transpose(img_input, (2, 0, 1))[np.newaxis, :, :, :]

    # 模型推論
    outputs = session.run(None, {input_name: img_input})
    detections = outputs[0]

    if len(detections.shape) == 3 and detections.shape[1] == 84:
        detections = detections[0].transpose((1, 0))  # (num_detections, 84)
    else:
        detections = detections[0]  # (num_detections, 84)

    boxes, confidences, class_ids = [], [], []
    for det in detections:
        score = float(det[4])
        if score < conf_threshold:
            continue

        x_center, y_center, w, h = det[0], det[1], det[2], det[3]
        x1 = int((x_center - w / 2 - pad_x) / scale)
        y1 = int((y_center - h / 2 - pad_y) / scale)
        x2 = int((x_center + w / 2 - pad_x) / scale)
        y2 = int((y_center + h / 2 - pad_y) / scale)
        cls_id = int(det[5])

        # 邊界保護
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])

        boxes.append([x1, y1, x2 - x1, y2 - y1])
        confidences.append(score)
        class_ids.append(cls_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        i = i[0] if isinstance(i, (tuple, list, np.ndarray)) else i
        x, y, w_box, h_box = boxes[i]
        cls_id = class_ids[i]
        score = confidences[i]

        label = f"{class_names[cls_id]} ({score:.2f})" if cls_id < len(class_names) else f"Unknown ({score:.2f})"
        print(f"{label}, Box: ({x},{y},{x+w_box},{y+h_box})")

        cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("YOLOv8 ONNX Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()