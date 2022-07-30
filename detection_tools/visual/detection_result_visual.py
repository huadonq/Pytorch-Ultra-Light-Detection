import json
import os
import cv2

if __name__=='__main__':
    parent_path = '../datasets/coco/images/val2017'
    json_file = 'coco_instances_val2017_results.json'
    with open(json_file) as annos:
        annotations = json.load(annos)

    for i in range(len(annotations)):
        annotation = annotations[i]
        if annotation['category_id'] != 11: #
            continue
        image_id = annotation['image_id']
        bbox = annotation['bbox'] # (x1, y1, w, h)
        x, y, w, h = bbox
        image_path = os.path.join(parent_path, str(image_id).zfill(12) + '.jpg') 
        image = cv2.imread(image_path)
        anno_image = cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 255), 2) 
        cv2.imshow('demo.jpg', anno_image)
 
