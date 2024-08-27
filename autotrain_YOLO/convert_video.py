import cv2
import os

video_path = r"E:\aHieu\autotrain_YOLO\Data\Muden_Aoxanh\1 (32).avi"

output_dir = r"E:\aHieu\autotrain_YOLO\Data\Muden_Aoxanh\1 (32)"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
count = 0
choose_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Khong phat hien frame")
        break

    frame_resize = cv2.resize(frame, (1024, 768))
    if count % 3 == 0:
        output_path = os.path.join(output_dir, "%04d.jpg" % choose_count)
        cv2.imwrite(output_path, frame_resize)
        choose_count += 1

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

    count += 1
    print(choose_count)
    # if choose_count == 20:
    #     break

cap.release()
cv2.destroyAllWindows()
