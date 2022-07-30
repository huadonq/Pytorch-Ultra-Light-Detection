import os
import cv2
from pycocotools.coco import COCO


if __name__=='__main__':
    json_file = '/home/jovyan/detection_tools/trainval.json'
    dataset_dir = '/home/jovyan/aaaaaaa'
    
    coco = COCO(json_file)
    catIds = coco.getCatIds(catNms=['11']) # catName
    imgIds = coco.getImgIds(catIds=catIds ) # catId
    for i in range(len(imgIds)):
        img = coco.loadImgs(imgIds[i])[0]
        image = cv2.imread(os.path.join(dataset_dir, img['file_name']))
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        annos = coco.loadAnns(annIds)

        bbox = annos[0]['bbox']
        x, y, w, h = bbox
        anno_image = cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 255), 5) 
        cv2.imwrite('demo.jpg', anno_image)
        exit()