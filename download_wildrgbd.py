# Modified from the original source
# Original source: https://github.com/wildrgbd/wildrgbd
# Original license: MIT License
# Copyright (c) 2024 rowdataset

import os 
import subprocess
import argparse
import random
import shutil


def prune_category(cat):
    cpath = os.path.join('dataset', cat)
    
    for s in os.listdir(os.path.join(cpath, 'scenes')):
        spath = os.path.join(cpath, 'scenes', s)
        indices = list(range(len(os.listdir(os.path.join(spath, 'depth')))))
        depths, rgbs = [sorted([os.path.join(spath, p, i) for i in os.listdir(os.path.join(spath, p))]) for p in ('depth', 'rgb')]
        
        with open(os.path.join(spath, 'cam_poses.txt'), 'r', encoding='utf8') as f:
            cam_poses = f.read().strip().split('\n')
        
        split = 30 # chooses n imgs
        random.shuffle(indices)
        indices, indices_del = sorted(indices[:split]), sorted(indices[split:])
        
        for p in (depths, rgbs):
            for i in indices_del:
                os.remove(p[i])
        
        shutil.rmtree(os.path.join(spath, 'masks'))
        
        cam_poses = [cam_poses[i] for i in indices]

        with open(os.path.join(spath, 'cam_poses.txt'), 'w', encoding='utf8') as f:
            f.write('\n'.join(cam_poses))


def download_category(path, categories, cat):
    cat_path = os.path.join(path, f'{cat}.zip')
    cat_single_path = os.path.join(path, f'{cat}-single.zip')

    for file in categories[cat]:
        subprocess.run(f'wget https://huggingface.co/hongchi/wildrgbd/resolve/main/{file}?download=true -O {file}', shell=True)

    if len(categories[cat]) > 1:
        subprocess.run(f'zip -F {cat_path} --out {cat_single_path}', shell=True)
        subprocess.run(f'unzip {cat_single_path} -d "{path}"', shell=True)
        subprocess.run(f'rm {cat_single_path}', shell=True)
        for file in categories[cat]:
            subprocess.run(f'rm {file}', shell=True)
    else:
        subprocess.run(f'unzip {cat_path} -d "{path}"', shell=True)
        subprocess.run(f'rm {cat_path}', shell=True)

    prune_category(cat)


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

    os.makedirs(path, exist_ok=True)
    categories_path = os.path.join(path, 'categories.txt')

    if cat == 'all':
        categories_list = sorted(list(categories.keys()))
        if os.path.exists(categories_path):
            with open(categories_path, 'r', encoding='utf8') as f:
                curr_cat = f.read()
            
            try:
                categories_list = categories_list[categories_list.index(curr_cat):]
            except ValueError:
                pass
        
        for cat in categories_list:
            print(f'\nDownloading `{cat}`...\n')
            with open(categories_path, 'w', encoding='utf8') as f:
                f.write(cat)

            download_category(path, categories, cat)
        
        os.remove(categories_path)
        print('Dataset downloaded')
                
    else:
        download_category(path, categories, cat)


if __name__ == '__main__':
    main()
