# 路径置顶
import sys 
import os 
sys.path.append(os.getcwd()) 
# 导入包
from tqdm import tqdm 
import numpy as np 
import dlib 
import cv2 

print('dlib.DLIB_USE_CUDA:', dlib.DLIB_USE_CUDA)
print('dlib.cuda.get_num_devices():', dlib.cuda.get_num_devices())

def preprocess(image_path, saveimg_path, savemask_path, detector, predictor, img_size):
    image = dlib.load_rgb_image(image_path)
    face_img, mask_img = None, None
    # 人脸对齐、切图
    dets = detector(image, 1)
    if len(dets) == 1:
        faces = dlib.full_object_detections()
        faces.append(predictor(image, dets[0]))
        images = dlib.get_face_chips(image, faces, size=img_size)

        image = np.array(images[0]).astype(np.uint8)
        face_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # 生成人脸mask
        dets = detector(image, 1)
        if len(dets) == 1:
            point68 = predictor(image, dets[0])
            landmarks = list()
            INDEX = [0,2,14,16,17,18,19,24,25,26]
            eyebrow_list = [19, 24]
            eyes_list = [36, 45]
            eyebrow = 0
            eyes = 0

            for eb, ey in zip(eyebrow_list, eyes_list):
                eyebrow += point68.part(eb).y
                eyes += point68.part(ey).y
            add_pixel = int(eyes/2 - eyebrow/2)

            for idx in INDEX:
                x = point68.part(idx).x
                if idx in eyebrow_list:
                    y = (point68.part(idx).y - 2*add_pixel) if (point68.part(idx).y - 2*add_pixel) > 0 else 0
                else:
                    y = point68.part(idx).y
                landmarks.append((x,y))
            landmarks = np.array(landmarks)
            hull = cv2.convexHull(landmarks)
            mask = np.zeros(face_img.shape, dtype=np.uint8)
            mask_img = cv2.fillPoly(mask, [hull], (255, 255, 255))

    if np.max(face_img) is not None and np.max(mask_img) is not None:
        cv2.imwrite(savemask_path, mask_img)
        cv2.imwrite(saveimg_path, face_img)



detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
img_size = 200

data_path = 'vggface2_train_250'
face_path = 'vggface2_train_250_face'
mask_path = 'vggface2_train_250_mask'
files_list = os.listdir(data_path)
for man in tqdm(files_list):
    pic_list = os.listdir(os.path.join(data_path, man))

    if not os.path.exists(os.path.join(mask_path, man)):
        os.mkdir(os.path.join(mask_path, man))
    if not os.path.exists(os.path.join(face_path, man)):
        os.mkdir(os.path.join(face_path, man))

    for pic in pic_list:
        img_path = os.path.join(data_path, man, pic)
        save_mask_path = os.path.join(mask_path, man, pic)
        save_face_path = os.path.join(face_path, man, pic)
        preprocess(img_path, save_face_path, save_mask_path, detector, predictor, img_size)
