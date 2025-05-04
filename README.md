# 基於 YOLOv8 ONNX 的即時物件偵測（Webcam）

本專案展示如何使用 YOLOv8 模型（經過 ONNX 格式轉換）進行即時物件偵測，透過攝影機畫面進行推論，並以 OpenCV 進行顯示與標註。

## 🔍 功能特色

- 支援即時攝影機畫面進行物件偵測
- 使用 letterbox 前處理保留影像比例，提升框選準確度
- 可自訂信心閾值與非極大值抑制（NMS）設定
- 偵測結果以「物件名稱」顯示（如 person、car 等）
- 輕量化實作，可在一般 CPU 或 Jetson 開發板上執行

## 🧰 環境依賴

請先安裝以下 Python 套件：

```bash
pip install onnxruntime opencv-python numpy



 💡 技術說明
使用 letterbox 將輸入影像縮放並加邊框，避免圖像變形，讓模型輸出座標更準確。

偵測結果包含框選邊界（bounding box）、類別名稱與信心分數。
