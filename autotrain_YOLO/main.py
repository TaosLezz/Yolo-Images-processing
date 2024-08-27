import os
os.environ['YOLO_VERBOSE'] = 'False'
from ultralytics import YOLO
import cv2

model = YOLO(r'E:\aHieu\autotrain_YOLO\best.pt')

video_path = r'E:\aHieu\autotrain_YOLO\Data\Muden_Aoxanh\1 (32).avi'

output_dir = r"E:\aHieu\autotrain_YOLO\output\Muden_Aoxanh\1 (32)"
os.makedirs(output_dir, exist_ok=True)


# Mở video đầu vào
cap = cv2.VideoCapture(video_path)

# Lấy thông tin về video đầu vào
count = 0
def convert_box_to_yoloform(frame, box, label):
    dh = 1. / frame.shape[0]
    dw = 1. / frame.shape[1]
    x = (int(box[0]) + int(box[2])) / 2.0
    y = (int(box[1]) + int(box[3])) / 2.0
    w = abs(int(box[2]) - int(box[0]))
    h = abs(int(box[3]) - int(box[1]))
    # ================
    x = round(x * dw, 6)
    w = round(w * dw, 6)
    y = round(y * dh, 6)
    h = round(h * dh, 6)
    sData = f'{label} {x} {y} {w} {h}\n'
    return sData

def save_txt(path='', filename='', data=None):
    if data and filename and path:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, filename), 'w') as w:
            for data in data:
                w.write(data)
            w.close()

while True:
    # Đọc từng khung hình từ video
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_resize = cv2.resize(frame, (1024, 768))
    im_process = frame_resize.copy()
    # Thực hiện dự đoán với YOLO
    results = model(frame_resize, conf=0.3)
    output_path_file = os.path.join(output_dir, f'frame{count}.jpg')
    output_path = os.path.join(output_dir, f"frame{count}.txt")
    labels_data = []
    
    for result in results:
        boxes = result.boxes  # Lấy bounding boxes
        for box in boxes:
            label = int(box.cls.item())  # Lấy nhãn (class)
            xyxy = box.xyxy.cpu().numpy()[0]  # Lấy tọa độ bounding box theo định dạng [x1, y1, x2, y2]
            x1, y1, x2, y2 = map(int, xyxy)

            # Vẽ bounding box lên khung hình
            cv2.rectangle(frame_resize, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_resize, f'Class {label}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if label != ' ':
                # sdata = convert_box_to_yoloform(frame, box=[x1, y1, x2, y2], label=label)
                dh = 1. / frame_resize.shape[0]
                dw = 1. / frame_resize.shape[1]
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1

                x_center = round(x_center * dw, 6)
                width = round(width * dw, 6)
                y_center = round(y_center * dh, 6)
                height = round(height * dh, 6)

                labels_data.append(f"{label} {x_center} {y_center} {width} {height}\n")
                with open(output_path, "w") as f:
                    
                    # f.write(f"{sdata}\n")
                    f.write(f"{label} {x_center} {y_center} {width} {height}\n")
    
    if labels_data:
        cv2.imwrite(output_path_file, im_process)

        # Write all labels data to the txt file
        with open(output_path, "w") as f:
            f.writelines(labels_data)
        count+=1
    # if count == 100:
    #     break
    print(count)
    # Hiển thị khung hình đã được chú thích (tùy chọn)
    cv2.imshow('YOLO Detection', frame_resize)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
