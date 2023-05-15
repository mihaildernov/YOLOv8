from ultralytics import YOLO

model = YOLO('yolov8m.pt')

if __name__ == '__main__':

    results = model.train(
        data='custom_data.yaml',
        imgsz=1920,
        epochs=3,
        batch=1,
        name='yolov8m_custom')
