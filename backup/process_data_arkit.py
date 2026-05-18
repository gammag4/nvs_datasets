import os
import sys
from easydict import EasyDict as edict
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
import random
import shutil


def get_frames(ipath, categories):
    categories_fnames = [
        ['.'.join(f.split('_')[-1].split('.')[:-1]) for f in os.listdir(os.path.join(ipath, c))]
        for c in categories
    ]
    
    with open(os.path.join(ipath, 'lowres_wide.traj'), 'r', encoding='utf8') as f:
        traj_lines = f.readlines()

    names = [str(round(float(l.split()[0]) * 1000) / 1000) for l in traj_lines]

    trajs = [[float(j) for j in i.split(' ')] for i in traj_lines]
    trajs = [edict(time=i[0], r=i[1:4], t=i[4:7]) for i in trajs]
    trajs = {k: v for k, v in zip(names, trajs)}

    names = set(names)
    for fnames in categories_fnames:
        names = names.intersection(fnames)

    frames = list(names)
    
    trajs = [trajs[f] for f in frames]
    
    return frames, trajs


def get_fxfycxcy_list(intrinsics_paths):
    fxfycxcy_list = []
    for p in intrinsics_paths:
        with open(p, 'r', encoding='utf8') as f:
            fxfycxcy = [float(i) for i in f.read().strip().split(' ')[2:]]
            fxfycxcy_list.append(fxfycxcy)
    
    return fxfycxcy_list


def get_w2c_list(trajs):
    rt_t_list = [(R.from_rotvec(np.array(t.r)).as_matrix().T, np.array(t.t).reshape(-1, 1)) for t in trajs]
    w2c_list = [np.concat([np.concat([Rt, -Rt @ t], axis=-1), np.array([[0, 0, 0, 1]])], axis=-2).tolist() for Rt, t in rt_t_list]
    
    return w2c_list


def process_item(path, item):
    ipath = os.path.join(path, item)

    if not os.path.isdir(ipath):
        return
    
    test_paths = [os.path.join(ipath, p) for p in ['lowres_depth', 'lowres_wide', 'lowres_wide_intrinsics', 'lowres_wide.traj']]
    for p in test_paths:
        if not (os.path.isdir(p) or os.path.isfile(p)):
            shutil.rmtree(ipath)
            return
    
    frames, trajs = get_frames(ipath, ['lowres_depth', 'lowres_wide', 'lowres_wide_intrinsics'])
    l = list(zip(frames, trajs))
    frames, trajs = zip(*random.sample(l, min(400, len(l)))) # randomly chooses 400 frames

    depth_fpath = os.path.join(ipath, 'lowres_depth')
    depth_paths = [os.path.join(depth_fpath, f'{item}_{f}.png') for f in frames]
    depth_paths_all = [os.path.join(depth_fpath, p) for p in os.listdir(depth_fpath)]
    for file in list(set(depth_paths_all) - set(depth_paths)):
        os.remove(file)

    pic_fpath = os.path.join(ipath, 'lowres_wide')
    pic_paths = [os.path.join(pic_fpath, f'{item}_{f}.png') for f in frames]
    pic_paths_all = [os.path.join(pic_fpath, p) for p in os.listdir(pic_fpath)]
    for file in list(set(pic_paths_all) - set(pic_paths)):
        os.remove(file)
    
    intrinsics_fpath = os.path.join(ipath, 'lowres_wide_intrinsics')
    intrinsics_paths = [os.path.join(intrinsics_fpath, f'{item}_{f}.pincam') for f in frames]
    intrinsics_paths_all = [os.path.join(intrinsics_fpath, p) for p in os.listdir(intrinsics_fpath)]
    for file in list(set(intrinsics_paths_all) - set(intrinsics_paths)):
        os.remove(file)
    
    fxfycxcy_list = get_fxfycxcy_list(intrinsics_paths)
    w2c_list = get_w2c_list(trajs)
    
    frames_list = [
        edict(
            image_path=pic,
            depth_path=depth,
            fxfycxcy=fxfycxcy,
            w2c=w2c
        )
        for depth, pic, fxfycxcy, w2c in zip(depth_paths, pic_paths, fxfycxcy_list, w2c_list)
    ]
    metadata = edict(
        scene_name=item,
        frames=frames_list
    )
    
    mpath = os.path.join(ipath, 'metadata.json')
    
    with open(mpath, 'w', encoding='utf8') as f:
        json.dump(metadata, f, indent=4)
