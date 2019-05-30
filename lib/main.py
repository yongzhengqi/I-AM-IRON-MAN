import torch
import torchvision.transforms as transforms
from .storehouse import mobilenet_v1
import numpy as np
import cv2
import os
import dlib
from .utils.ddfa import ToTensorGjz, NormalizeGjz
import scipy.io as sio
from .utils.inference import parse_roi_box_from_landmark, crop_img, predict_68pts, draw_landmarks, predict_dense, \
    calc_hypotenuse
from .utils.cv_plot import plot_pose_box, build_camera_box
from .utils.estimate_pose import parse_pose
from .utils.render import cget_depths_image, cpncc
import math
import pyrender
from tqdm import tqdm

STD_SIZE = 120


def gen_rotate_matrix(roll, pitch, yaw):
    roll, pitch, yaw = pitch, yaw, roll

    yawMatrix = np.matrix([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]
    ])

    pitchMatrix = np.matrix([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]
    ])

    rollMatrix = np.matrix([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll), math.cos(roll)]
    ])

    R = yawMatrix * pitchMatrix * rollMatrix

    R = np.vstack([R, [0, 0, 0]])
    a = np.array([[0], [0], [0], [1]])
    R = np.hstack([R, a])

    return R


def rendering(R, fuze_trimesh):
    mesh = pyrender.Mesh.from_trimesh(fuze_trimesh, poses=[R])
    scene = pyrender.Scene()
    scene.add(mesh)
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3, aspectRatio=1)
    m1 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype='float')
    m2 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0.4],
        [0, 0, 0, 1],
    ], dtype='float')
    camera_pose = m1.dot(m2)

    scene.add(camera, pose=camera_pose)
    light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
                               innerConeAngle=np.pi / 16.0)
    scene.add(light, pose=camera_pose)
    r = pyrender.OffscreenRenderer(400, 400)
    color, depth = r.render(scene)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

    return color


def add_mask(front, back, mat):
    front_tra = cv2.warpAffine(front, mat, (back.shape[1], back.shape[0]),
                               np.zeros(back.shape),
                               borderValue=(255, 255, 255))
    mask = np.sum(front_tra, axis=2)
    mask = np.stack([mask, mask, mask], axis=2)
    dst = np.where(mask == 255 * 3, back, front_tra)
    return dst


def magic_func(img_ori, poses, mats, fuze_trimesh):
    img_magic = img_ori
    for pose, mat in zip(poses, mats):
        R = gen_rotate_matrix(pose[2], pose[1], -pose[0])
        mask = rendering(R, fuze_trimesh)
        img_magic = add_mask(mask, img_magic, mat)

    return img_magic


def get_2D_points(Ps, pts68s, Pts=[150, 150, 240, 230]):
    matrice = []
    if not isinstance(pts68s, list):
        pts68s = [pts68s]
    if not isinstance(Ps, list):
        Ps = [Ps]
    for i in range(len(pts68s)):
        pts68 = pts68s[i]
        llength = calc_hypotenuse(pts68)
        point_3d = build_camera_box(llength)
        P = Ps[i]

        # Map to 2d image points
        point_3d_homo = np.hstack((point_3d, np.ones([point_3d.shape[0], 1])))  # n x 4
        point_2d = point_3d_homo.dot(P.T)[:, :2]

        point_2d[:, 1] = - point_2d[:, 1]
        point_2d[:, :2] = point_2d[:, :2] - np.mean(point_2d[:4, :2], 0) + np.mean(pts68[:2, :27], 1)
        point_2d = np.int32(point_2d.reshape(-1, 2))

        src = np.float32([[Pts[0], Pts[1]], [Pts[2], Pts[1]], [Pts[0], Pts[3]]])
        dst = np.float32([point_2d[1], point_2d[2], point_2d[0]])
        matrix = cv2.getAffineTransform(src, dst)
        matrice.append(matrix)

    return matrice


def magic(img_ori, fuze_trimesh, args):
    # 1. load pre-tained model
    checkpoint_fp = args.mobilenet_path
    arch = 'mobilenet_1'

    checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
    model = getattr(mobilenet_v1, arch)(num_classes=62)  # 62 = 12(pose) + 40(shape) +10(expression)

    model_dict = model.state_dict()
    # because the model is trained by multiple gpus, prefix module should be removed
    for k in checkpoint.keys():
        model_dict[k.replace('module.', '')] = checkpoint[k]
    model.load_state_dict(model_dict)
    model.eval()

    # 2. load dlib model for face detection and landmark used for face cropping
    dlib_landmark_model = args.dlib_model_path
    face_regressor = dlib.shape_predictor(dlib_landmark_model)
    face_detector = dlib.get_frontal_face_detector()

    # 3. forward
    transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])

    rects = face_detector(img_ori, 1)

    pts_res = []
    Ps = []  # Camera matrix collection
    poses = []  # pose collection
    vertices_lst = []  # store multiple face vertices
    ind = 0
    for rect in rects:
        # whether use dlib landmark to crop image, if not, use only face bbox to calc roi bbox for cropping
        # - use landmark for cropping
        pts = face_regressor(img_ori, rect).parts()
        pts = np.array([[pt.x, pt.y] for pt in pts]).T
        roi_box = parse_roi_box_from_landmark(pts)

        img = crop_img(img_ori, roi_box)

        # forward: one step
        img = cv2.resize(img, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
        input = transform(img).unsqueeze(0)
        with torch.no_grad():
            param = model(input)
            param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

        # 68 pts
        pts68 = predict_68pts(param, roi_box)

        pts_res.append(pts68)
        P, pose = parse_pose(param)
        Ps.append(P)
        poses.append(pose)

        # dense face 3d vertices
        vertices = predict_dense(param, roi_box)
        vertices_lst.append(vertices)

        ind += 1

    # P, pose = parse_pose(param)  # Camera matrix (without scale), and pose (yaw, pitch, roll, to verify)
    img_pose = plot_pose_box(img_ori, Ps, pts_res)
    # img_pose = cv2.cvtColor(img_pose, cv2.COLOR_BGR2RGB)

    landmark_img = draw_landmarks(img_ori, pts_res)

    mats = get_2D_points(Ps, pts_res)
    img_magic = magic_func(img_ori, poses, mats, fuze_trimesh)

    img_landmark = cv2.cvtColor(landmark_img, cv2.COLOR_BGR2RGB)
    img_pose = cv2.cvtColor(img_pose, cv2.COLOR_BGR2RGB)
    img_magic = cv2.cvtColor(img_magic, cv2.COLOR_BGR2RGB)

    return img_pose, img_landmark, img_magic
