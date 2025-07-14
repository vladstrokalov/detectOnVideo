# Проверка качества обучения сети YOLO / YOLO Inference Quality Checker

Графическое приложение для проверки качества работы нейросети YOLO на видеофайлах.

A graphical application for evaluating the performance and quality of YOLO neural networks on video files.

---

## 📋 Требования / Requirements

- **Qt 6**
- **OpenCV 4.x** with **CUDA** support
- **ONNX Runtime**

---

## ⚙️ Сборка / Build

```bash
git clone git@github.com:vladstrokalov/detectOnVideo.git
cd detectOnVideo
mkdir build
cd build
cmake ..
make
```
---

## 🚀 Использование / Usage

### RU:
1. Загрузить обученную нейросеть (ONNX).
2. Установить параметры и путь сохранения результатов (при необходимости).
3. Загрузить видеофайл.
4. Использовать кнопки **Start / Stop** для запуска или остановки распознавания.

### EN:
1. Load a trained neural network (ONNX).
2. Set recognition parameters and output path (if needed).
3. Load a video file.
4. Use **Start / Stop** buttons to begin or stop inference.

---

## 🧠 О проекте / About

Проект предназначен для визуальной оценки распознавания объектов с помощью сети YOLO на видеоматериалах. Поддерживается ускорение через CUDA.

This project provides a visual tool for testing YOLO object detection on video files, with CUDA acceleration supported.

