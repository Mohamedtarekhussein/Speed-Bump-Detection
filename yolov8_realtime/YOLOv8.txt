import cv2
from ultralytics import YOLO

video_path = 'bump5.mp4'

# Increase the value of img_size for faster processing
model = YOLO('best3.pt')

cap = cv2.VideoCapture(video_path)

frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter('video5_out.mp4',fourcc,30,(frame_width,frame_height))

while cap.isOpened():
    success, frame = cap.read()
    if success:
        results = model.predict(frame,conf= 0.3 ,iou = 0.4)
        annotated_frame = results[0].plot()
        #cv2.imshow("YOLOv8 Inference", annotated_frame)
        video_writer.write(annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break
cap.release()

# Destroy all OpenCV windows
cv2.destroyAllWindows()