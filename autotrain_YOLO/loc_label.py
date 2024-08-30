import os

file_dir = r'E:\aHieu\autotrain_YOLO\output\Muden_AoTrang\1 (46)'


for file_name in os.listdir(file_dir):
    if file_name.startswith('frame') and file_name.endswith('.txt'):
        file_path = os.path.join(file_dir, file_name)

        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        filtered_line = [line for line in lines if line.startswith(('2', '5'))]

        with open(file_path, 'w') as f:
            # print(filtered_line)
            f.writelines(filtered_line)
            print("done")