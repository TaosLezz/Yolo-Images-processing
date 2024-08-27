# from ultralytics import YOLO
# import cv2
# import os

# model = YOLO(r'E:\aHieu\autotrain_YOLO\best.pt')

# video_path = r'E:\aHieu\autotrain_YOLO\videos\meo.mp4'

# output_dir = r"E:\aHieu\autotrain_YOLO\output"
# os.makedirs(output_dir, exist_ok=True)
# image_width = 640 
# image_height = 640 
# # Mở video đầu vào
# cap = cv2.VideoCapture(video_path)

# count = 0

# while True:
#     # Đọc từng khung hình từ video
#     ret, frame = cap.read()
#     im_process = frame.copy()
#     if not ret:
#         break

#     # Thực hiện dự đoán với YOLO
#     results = model(im_process, conf=0.6)
#     output_path_file = os.path.join(output_dir, f'frame{count}.jpg')
#     for result in results:
#         boxes = result.boxes  # Lấy bounding boxes
#         for box in boxes:
#             label = int(box.cls.item())  # Lấy nhãn (class)
#             xyxy = box.xyxy.cpu().numpy()[0]  # Lấy tọa độ bounding box theo định dạng [x1, y1, x2, y2]
#             x1, y1, x2, y2 = map(int, xyxy)
            
#             # Vẽ bounding box lên khung hình
#             cv2.rectangle(im_process, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(im_process, f'Class {label}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#             if label == 0:
#                 cv2.imwrite(output_path_file, im_process)
#                 output_path = os.path.join(output_dir, f"frame{count}.txt")
#                 count+=1
#                 with open(output_path, "w") as f:
#                     for result in results:
#                         boxes = result.boxes  # Lấy bounding boxes
#                         for box in boxes:
#                             label = int(box.cls.item())  # Lấy nhãn (class)
#                             xyxy = box.xyxy.cpu().numpy()[0]  # Lấy tọa độ bounding box theo định dạng [x1, y1, x2, y2]
#                             # Tính toán tọa độ tâm và kích thước bounding box
#                             x1, y1, x2, y2 = map(int, xyxy)
#                             x_center = (x1 + x2) / 2
#                             y_center = (y1 + y2) / 2
#                             width = x2 - x1
#                             height = y2 - y1

#                             # Chuyển đổi tọa độ về tỷ lệ phần trăm
#                             x_center /= image_width
#                             y_center /= image_height
#                             width /= image_width
#                             height /= image_height
                        
#                             f.write(f"{label} {x_center} {y_center} {width} {height}\n")

#     # Hiển thị khung hình đã được chú thích (tùy chọn)
#     cv2.imshow('YOLO Detection', im_process)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Giải phóng tài nguyên
# cap.release()
# # out.release()
# cv2.destroyAllWindows()
from ultralytics import YOLO
import cv2
import os

model = YOLO(r'E:\aHieu\autotrain_YOLO\best.pt')

video_path = r'E:\aHieu\autotrain_YOLO\videos\meo.mp4'

output_dir = r"E:\aHieu\autotrain_YOLO\output"
os.makedirs(output_dir, exist_ok=True)
image_width = 640 
image_height = 640 

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
image_files = [f for f in os.listdir(output_dir) if f.endswith('.jpg')]
for image_file in image_files:
    # Đọc ảnh từ thư mục
    image_path = os.path.join(output_dir, image_file)
    image = cv2.imread(image_path)
    
    # Thực hiện dự đoán với YOLO
    results = model(image, conf=0.6)
    txt_file_path = os.path.join(output_dir, image_file.replace('.jpg', '.txt'))
    
    with open(txt_file_path, 'w') as f:
        for result in results:
            boxes = result.boxes
            for box in boxes:
                label = int(box.cls.item())
                xyxy = box.xyxy.cpu().numpy()[0]
                x1, y1, x2, y2 = xyxy
                
                # Chuyển đổi bounding box thành định dạng YOLO
                sdata = convert_box_to_yoloform(image, box=[x1, y1, x2, y2], label=label)
                f.write(f"{sdata}\n")

print("Dự đoán hoàn tất và kết quả đã được lưu.")
