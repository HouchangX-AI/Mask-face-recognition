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

def add_extension(path):
    # 文件格式比较迷，可能有这两种情况
    if os.path.exists(path + '.jpg'):
        return path + '.jpg'
    elif os.path.exists(path + '.png'):
        return path + '.png'
    else:
        raise RuntimeError('No file "%s" with extension png or jpg.' % path)

def preprocess(image_path, face_path, mask_path, detector, predictor, img_size):
    image = dlib.load_rgb_image(image_path)
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

    cv2.imwrite(face_path, face_img)
    cv2.imwrite(mask_path, mask_img)

root_dir = 'vggface2_train_250'
face_path = 'vggface2_train_250_face'
mask_path = 'vggface2_train_250_mask'
training_triplets_path = 'training_triplets_100000.npy'
training_triplets = np.load(training_triplets_path)

max_idx = len(training_triplets)
for index in tqdm(range(max_idx)):
    anc_id, pos_id, neg_id, pos_class, neg_class, pos_name, neg_name = training_triplets[index]

    anc_img = add_extension(os.path.join(root_dir, str(pos_name), str(anc_id)))
    pos_img = add_extension(os.path.join(root_dir, str(pos_name), str(pos_id)))
    neg_img = add_extension(os.path.join(root_dir, str(neg_name), str(neg_id)))

    if not os.path.exists(os.path.join(face_path, str(pos_name))):
        os.mkdir(os.path.join(face_path, str(pos_name)))
    if not os.path.exists(os.path.join(mask_path, str(pos_name))):
        os.mkdir(os.path.join(face_path, str(pos_name)))
    if not os.path.exists(os.path.join(face_path, str(neg_name))):
        os.mkdir(os.path.join(face_path, str(neg_name)))
    if not os.path.exists(os.path.join(mask_path, str(neg_name))):
        os.mkdir(os.path.join(face_path, str(neg_name)))

    face_man_path = os.path.join(face_path, str(pos_name), str(anc_id)) + '.jpg'
    mask_man_path = os.path.join(mask_path, str(pos_name), str(anc_id)) + '.jpg'




    preprocess(anc_img, detector, predictor, img_size)
    preprocess(pos_img, detector, predictor, img_size)
    preprocess(neg_img, detector, predictor, img_size)


