# Modified from the original source
# Original source: https://github.com/wildrgbd/wildrgbd
# Original license: MIT License
# Copyright (c) 2024 rowdataset

import os
import subprocess
import argparse
import random
import shutil


def select_views(rgbs, depths, cam_poses, num_views, view_cone_range=(None, None)):
    view_cone_range = [len(rgbs) - 1 if v is None else v for v in view_cone_range]
    
    items = list(zip(rgbs, depths, cam_poses))
    sources = [items[v % len(rgbs)] for v in view_cone_range]
    
    view_cone_range = (view_cone_range[0] + 1) % len(rgbs), (view_cone_range[1] - 1) % len(rgbs)
    
    if view_cone_range[1] < view_cone_range[0]:
        targets = items[view_cone_range[0]:] + items[:view_cone_range[1]]
    else:
        targets = items[view_cone_range[0]:view_cone_range[1]]
    targets = list(enumerate(targets))
    targets = random.sample(targets, num_views - 2)
    targets.sort(key=lambda x: x[0])
    targets = [e for _, e in targets]
    
    items = [sources[0]] + targets + [sources[1]]
    return list(zip(*items))


def process_cone(cpath, cone):
    rgbs, depths, cam_poses = cone
    
    for r, d in zip(rgbs, depths):
        for p, c in ((r, 'rgb'), (d, 'depth')):
            rpath = os.path.join(cpath, c)
            os.makedirs(rpath, exist_ok=True)
            rpath = os.path.join(rpath, os.path.split(p)[1])
            shutil.copy2(p, rpath)
    
    with open(os.path.join(cpath, 'cam_poses.txt'), 'w', encoding='utf8') as f:
        f.write('\n'.join(cam_poses))


def process_scene(spath, num_view_cones, view_cone_range, num_views):
    print(f'Processing scene  "{spath}"...')
    rgb_path, depth_path, cam_poses_path = [os.path.join(spath, p) for p in ('rgb', 'depth', 'cam_poses.txt')]
    
    if [os.path.exists(p) for p in (rgb_path, depth_path, cam_poses_path)] != [True, True, True]:
        print(f'Inconsistency in dataset paths found at "{spath}", skipping scene')
        shutil.rmtree(spath)
        return False
    
    rgbs, depths = [sorted([os.path.join(spath, p, i) for i in os.listdir(p)]) for p in (rgb_path, depth_path)]
    with open(cam_poses_path, 'r', encoding='utf8') as f:
        cam_poses = f.read().strip().split('\n')
    
    indices, depth_indices = [sorted([int(i.split('.')[0]) for i in os.listdir(p)]) for p in (rgb_path, depth_path)]
    cam_pose_indices = [int(l.split()[0]) for l in cam_poses]
    
    if not (indices == depth_indices == cam_pose_indices):
        print(f'Inconsistency in dataset paths found at "{spath}", skipping scene')
        shutil.rmtree(spath)
        return False
    
    cone_indices = random.sample(indices, num_view_cones)
    cone_indices = [(i, (i + random.randint(*view_cone_range))) for i in cone_indices]
    cones = [select_views(rgbs, depths, cam_poses, num_views, v) for v in cone_indices]
    
    for i, cone in enumerate(cones):
        cpath = os.path.join(spath, 'cones', str(i))
        process_cone(cpath, cone)
    
    shutil.rmtree(rgb_path)
    shutil.rmtree(depth_path)
    os.remove(cam_poses_path)
    shutil.rmtree(os.path.join(spath, 'masks'))
    
    return True


def process_category(cpath, num_view_cones, view_cone_range, num_views):
    failed = []
    for s in os.listdir(os.path.join(cpath, 'scenes')):
        spath = os.path.join(cpath, 'scenes', s)
        success = process_scene(spath, num_view_cones, view_cone_range, num_views)
        if not success:
            failed.append(spath)
    
    if len(failed) > 0:
        with open(os.path.join(cpath, 'failed_scenes.txt'), 'w', encoding='utf8') as f:
            f.write('\n'.join(failed))


def download_category(path, categories, cat, num_view_cones, view_cone_range, num_views):
    cpath = os.path.join(path, cat)
    cat_path = os.path.join(path, f'{cat}.zip')
    cat_single_path = os.path.join(path, f'{cat}-single.zip')
    
    if not os.path.exists(cat_single_path):
        for file in categories[cat]:
            fpath = os.path.join(path, file)
            subprocess.run(f'wget -c https://huggingface.co/hongchi/wildrgbd/resolve/main/{file}?download=true -O {fpath}', shell=True)
    
    if len(categories[cat]) > 1:
        subprocess.run(f'zip -F {cat_path} --out {cat_single_path}', shell=True)
        for file in categories[cat]:
            fpath = os.path.join(path, file)
            subprocess.run(f'rm {fpath}', shell=True)
        subprocess.run(f'unzip {cat_single_path} -d "{path}"', shell=True)
        subprocess.run(f'rm {cat_single_path}', shell=True)
    else:
        subprocess.run(f'unzip {cat_path} -d "{path}"', shell=True)
        subprocess.run(f'rm {cat_path}', shell=True)
        
    process_category(cpath, num_view_cones, view_cone_range, num_views)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--cat", type=str, required=True)
    args = parser.parse_args()
    
    cat = args.cat
    path = args.path
    
    categories = {
    'bottle': ['bottle.z01', 'bottle.z02', 'bottle.zip'],
    'cup': ['cup.z01', 'cup.z02', 'cup.z03', 'cup.zip'],
    'tooth_brush': ['tooth_brush.z01', 'tooth_brush.z02', 'tooth_brush.z03', 'tooth_brush.zip'],
    'book': ['book.z01', 'book.zip'],
    'box': ['box.z01', 'box.zip'],
    'knife': ['knife.z01', 'knife.z02', 'knife.zip'],
    'remote_control': ['remote_control.z01', 'remote_control.zip'],
    'razor': ['razor.z01', 'razor.zip'],
    'keyboard': ['keyboard.z01', 'keyboard.zip'],
    'bowl': ['bowl.z01', 'bowl.z02', 'bowl.zip'],
    'scissor': ['scissor.z01', 'scissor.z02', 'scissor.z03', 'scissor.zip'],
    'kettle': ['kettle.z01', 'kettle.z02', 'kettle.zip'],
    'bucket': ['bucket.zip'],
    'pliers': ['pliers.z01', 'pliers.z02', 'pliers.zip'],
    'ball': ['ball.z01', 'ball.zip'],
    'mouse': ['mouse.z01', 'mouse.zip'],
    'handbag': ['handbag.z01', 'handbag.z02', 'handbag.zip'],
    'cellphone': ['cellphone.zip'],
    'microwave': ['microwave.z01', 'microwave.zip'],
    'hat': ['hat.z01', 'hat.z02', 'hat.zip'],
    'clock': ['clock.z01', 'clock.zip'],
    'shoe': ['shoe.z01', 'shoe.z02', 'shoe.z03', 'shoe.zip'],
    'flower_pot': ['flower_pot.z01', 'flower_pot.z02', 'flower_pot.zip'],
    'detergent': ['detergent.z01', 'detergent.z02', 'detergent.zip'],
    'backpack': ['backpack.z01', 'backpack.z02', 'backpack.zip'],
    'chair': ['chair.z01', 'chair.z02', 'chair.zip'],
    'TV': ['TV.zip'],
    'pineapple': ['pineapple.z01', 'pineapple.zip'],
    'potato': ['potato.zip'],
    'cucumber': ['cucumber.zip'],
    'apple': ['apple.zip'],
    'banana': ['banana.zip'],
    'pear': ['pear.zip'],
    'tomato': ['tomato.zip'],
    'peach': ['peach.zip'],
    'orange': ['orange.z01', 'orange.zip'],
    'carrot': ['carrot.zip'],
    'donut': ['donut.zip'],
    'cake': ['cake.zip'],
    'stuffed_toy': ['stuffed_toy.z01', 'stuffed_toy.z02', 'stuffed_toy.zip'],
    'train': ['train.zip'],
    'truck': ['truck.zip'],
    'boat': ['boat.zip'],
    'bus': ['bus.zip'],
    'plane': ['plane.zip'],
    'car': ['car.zip'],
    }
    categories_list = sorted(list(categories.keys()))
    
    assert cat == 'all' or categories.get(cat, False), f'Invalid category "{cat}"'
    
    # TODO
    # Just randomly samples views
    # num_view_cones = 1
    # view_cone_range = (None, None)
    # num_views = 60
    
    # Creates multiple cone samples
    num_view_cones = 6
    view_cone_range = (40, 70) # min and max sizes for cone range
    num_views = 10 # num views per cone
    
    os.makedirs(path, exist_ok=True)
    download_progress_path = os.path.join(path, 'download_progress.txt')
    
    if cat == 'all':
        if os.path.exists(download_progress_path):
            with open(download_progress_path, 'r', encoding='utf8') as f:
                curr_cat = f.read()
            
            try:
                categories_list = categories_list[categories_list.index(curr_cat):]
            except ValueError:
                pass
        
        for cat in categories_list:
            print(f'\nDownloading category "{cat}"...\n')
            with open(download_progress_path, 'w', encoding='utf8') as f:
                f.write(cat)
            
            download_category(path, categories, cat, num_view_cones, view_cone_range, num_views)
        
        os.remove(download_progress_path)
    else:
        download_category(path, categories, cat, num_view_cones, view_cone_range, num_views)
    
    failed = []
    for cat in categories_list:
        failed_path = os.path.join(path, cat, 'failed_scenes.txt')
        if os.path.isfile(failed_path):
            with open(failed_path, 'r', encoding='utf8') as f:
                failed.extend(f.read().split('\n'))
    
    if len(failed) > 0:
        print('\nScenes that failed:')
        for s in failed:
            print(s)
        
        with open(os.path.join(path, 'failed.txt'), 'w', encoding='utf8') as f:
            f.write('\n'.join(failed))
    
    print('\nDataset downloaded')


if __name__ == '__main__':
    main()

# Usage: python download_wildrgbd.py --cat all --path "dataset/destination/path"
