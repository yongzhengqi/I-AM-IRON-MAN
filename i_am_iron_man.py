import cv2
from lib.main import magic
from tqdm import tqdm
import numpy as np
import trimesh
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--helmet', dest='helmet_model_path', required=False, default=os.path.join(os.getcwd(), 'lib/models/Mark 42.obj'))
    parser.add_argument('-m', '--mobile', dest='mobilenet_path', required=False, default=os.path.join(os.getcwd(), 'lib/models/phase1_wpdc_vdc.pth.tar'))
    parser.add_argument('-d', '--dlib', dest='dlib_model_path', required=False, default=os.path.join(os.getcwd(), 'lib/models/shape_predictor_68_face_landmarks.dat'))
    args = parser.parse_args()

    fuze_trimesh = trimesh.load(args.helmet_model_path)

    for idx, vertix in tqdm(enumerate(fuze_trimesh.vertices), total=3251, desc='loading 3D model'):
        if vertix[1] < 1.57:
            tmp = list()
            for face in fuze_trimesh.faces:
                if face[0] != idx and face[1] != idx and face[2] != idx:
                    tmp.append(face)
            fuze_trimesh.faces = np.array(tmp)
        else:
            vertix[1] -= 1.67
            vertix[2] -= 0.06

    print('Press \'q\' to quite')

    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()

        frame = cv2.resize(frame, (640, 360))

        img_pose, img_landmark, img_final = magic(frame, fuze_trimesh, args)

        img_pose = cv2.resize(img_pose, (640, 360))
        img_landmark = cv2.resize(img_landmark, (640, 360))
        img_final = cv2.resize(img_final, (640, 360))
        img1 = np.concatenate([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), img_landmark], axis=0)
        img2 = np.concatenate([img_pose, img_final], axis=0)
        img = np.concatenate([img1, img2], axis=1)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow('I am Iron Man.', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
