"""
input: coco json file
output: 
1.train.txt
 list of train image paths
2. val.txt
 list of validation image paths
3.  test.txt
 list of test image paths
4. bunch of label.txt
 file name: image name
 content: center_x, center_y, width, height, box_conf, class_conf, kp_1_x, kp_1_y, kp_1_vis, ....

    1. make map of  "image id -> [annotation]"
    2. per image id, make a file with name: "image_file_name.txt" and content box1 \n box2 \n box3, .....
"""

import json, os

json_path = "/home/min/Documents/yolov7_custom/my_mouse/my_mouse-2.json"
yolo_file_output_dir = "/home/min/Documents/yolov7_custom/yolov7/yolo_data"

is_train_set = False
def main():
    
    with open(json_path, 'r') as json_file:
        json_data = json.load(json_file)

        # 1. image file list .txt
        images = json_data['images']
        # print(images)
        image_width = images[0]['width']
        image_height = images[0]['height']

        image_id_map = dict()
        image_file_names = []
        for img in images:
            image_id_map[img['id']] = img
            print(img['id'])
            image_file_names.append(img['file_name'])
        

        # print(" ")
        # print(image_file_names)

        yolo_output = yolo_file_output_dir
        k = 1

        # while os.path.exists(yolo_output):
        #     yolo_output = yolo_file_output_dir + '_' + str(k)
        #     k += 1
        # print('YOLO dataset output path: ' + yolo_output)
        # os.mkdir(yolo_output)
        
        # 1. images path list txt
        
        image_list_file_path = os.path.join(yolo_output, 'train.txt') if is_train_set else os.path.join(yolo_output, 'test.txt')
        
        images_dir = './images/train' if is_train_set else './images/test'
        
        with open(image_list_file_path, 'w') as images_file:
            for img in image_file_names:
                images_file.write(str(os.path.join(images_dir,img)) + '\n')


        # 2. labels
        annotations = json_data['annotations']
        annotations_map = dict()
        
        for annotation in annotations:
            id = annotation['image_id']
            if not id in annotations_map:
                annotations_map[id] = [annotation]
            else:
                annotations_map[id].append(annotation)
        
        labels_path = os.path.join(yolo_output, "labels/")
        if os.path.exists(labels_path):
            print("label path ", labels_path, 'already exists. overwrite.')
        else:
            os.mkdir(labels_path)

        save_path = os.path.join(labels_path + 'train/') if is_train_set else os.path.join(labels_path + 'test')
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        for id in annotations_map.keys():
            image_file_name = image_id_map[id]['file_name']
            label_file_name = image_file_name[:-3] + 'txt'
            annotation_strings = []
            for ann in annotations_map[id]:
                # 1. bounding box normalization
                bbox = ann['bbox']
                c_x = (bbox[0] + bbox[2]/2) / image_width
                c_y = (bbox[1] + bbox[3]/2) / image_height
                normed_width = bbox[2] / image_width
                normed_height = bbox[3] / image_height
                annotation_string = f'0 {c_x} {c_y} {normed_width} {normed_height}'
                keypoints = ann['keypoints']
                num_kps = ann['num_keypoints']
                
                for i in range(num_kps):
                    annotation_string += f' {keypoints[3*i]/image_width} {keypoints[3*i+1]/image_height} {keypoints[3*i+2]}'
                annotation_strings.append(annotation_string)

            with open(os.path.join(save_path, label_file_name), 'w') as label_file:
                label_file.writelines(annotation_strings)
                print("wrote " + label_file_name)

                    
            



if __name__ == "__main__":
    main()