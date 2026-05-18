import json
import asyncio
import aiohttp
from matplotlib import pyplot as plt
import random
import os
from urllib.request import urlretrieve
import progressbar


class MyProgressBar():
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = progressbar.ProgressBar(maxval=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()


class PanopticScene:
    def __init__(self, scene_name):
        self.scene_name = scene_name
        self.url = f'http://domedb.perception.cs.cmu.edu/webdata/dataset/{scene_name}'
        self.video_url = {
            'hd': f'{self.url}/videos/hd_shared_crf20',
            'vga': f'{self.url}/videos/vga_shared_crf10',
            'kinect-color': ''
        }
        self.fps = {'hd': 29.97, 'vga': 25.0, 'kinect-color': 30}

        self.cameras = None

    async def load(self):
        cameras = await self._get_cameras()
        cameras = [self._process_camera(c) for c in cameras]

        self.cameras = {t: list(filter(lambda x: x['type'] == t, cameras)) for t in [
            'hd', 'vga']}

    async def _get_cameras(self):
        async with aiohttp.ClientSession() as session:
            async with session.get(f'{self.url}/calibration_{self.scene_name}.json') as resp:
                return (await resp.json())['cameras']

    def _process_camera(self, c):
        cam_type = c['type']
        cam_name = c['name']
        cam_fname = f'{cam_type}_{cam_name}.mp4'
        
        c['fps'] = self.fps[cam_type]
        c['url'] = f'{self.video_url[cam_type]}/{cam_fname}'
        c['filename'] = cam_fname
        
        return {k: c[k] for k in ['filename', 'type', 'url', 'K', 'R', 't', 'fps', 'distCoef']}
        # return c

    def get_views(self, cam_type):
        return self.cameras[cam_type].copy()


class PanopticDownloader:
    def __init__(self, device, progress_file=None):
        self.scenes = None
        self.device = device
        self.progress = {}

        if progress_file is not None:
            self.progress_file = progress_file
            with open(progress_file) as f:
                self.progress = json.load(f)

    async def load(self, scene_names_file):
        scenes = [PanopticScene(n) for n in self._get_scene_names(scene_names_file)]

        bsize = 3
        batches = [scenes[i:i+bsize] for i in range(0, len(scenes), bsize)]
        for batch in progressbar.progressbar(batches):
            await asyncio.gather(*[s.load() for s in batch])

        self.scenes = scenes

    def _get_scene_names(self, scene_names_file):
        with open(scene_names_file, encoding='utf-8') as f:
            return [i.strip() for i in f.readlines()]

    def download_views(self, path, cam_type, n_views, n_scenes=None, stop_after_n_fails=None):
        print(f'Downloading scenes...')

        scenes = self.scenes.copy()
        if n_scenes is None:
            n_scenes = -1
        else:
            random.shuffle(scenes)
        
        scenes_downloaded = self.progress.get('scenes_downloaded', [])
        scenes_failed = self.progress.get('scenes_failed', [])
        scenes_tried = self.progress.get('scenes_tried', [])
        i = self.progress.get('i', 0)
        while len(scenes) > 0:
            s = scenes[0]
            
            n_views_downloaded = self.progress.get('n_views_downloaded', 0)
            fails = self.progress.get('fails', 0)
            views = s.get_views(cam_type) self.progress.get('fails', 0)
            scene = (s.scene_name, [])

            with open(os.path.join(path, self.progress_file), 'w', encoding='utf-8') as f:
                self.progress = {
                    'scenes': scenes,
                    'n_scenes': n_scenes,
                    'scenes_downloaded': scenes_downloaded,
                    'scenes_failed': scenes_failed,
                    'scenes_tried': scenes_tried,
                    'n_views_downloaded': n_views_downloaded
                }
                json.dump(self.progress, f, indent=2, sort_keys=True)
            
            j = 0
            while len(views) > 0:
                v = views[0]
                
                ns = min(n_scenes, len(scenes))
                nv = min(n_views, len(views))
                print(f"Scene {len(scenes_downloaded)}/{ns}, view {n_views_downloaded}/{nv}, url {v['url']}")
                
                p = os.path.join(path, s.scene_name)
                os.makedirs(p, exist_ok=True)
                p = os.path.join(p, v['filename'])

                success = self.try_downloading_view(v['url'], p)

                if success:
                    print('\tSuccess')
                    scene[1].append(v)
                    fails = 0
                    n_views_downloaded += 1
                    if n_views_downloaded == n_views:
                        scenes_downloaded.append(scene)
                        break
                    continue
                
                fails += 1
                print('\tFailed')

                if stop_after_n_fails is not None and fails >= stop_after_n_fails:
                    if n_views_downloaded > 0:
                        scenes_failed.append(scene)
                    break
                
                views.pop(0)
            
            scenes.pop(0)

        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'complete.json'), 'w', encoding='utf-8') as f:
            json.dump(scenes_downloaded, f, indent=2, sort_keys=True)
        with open(os.path.join(path, 'incomplete.json'), 'w', encoding='utf-8') as f:
            json.dump(scenes_failed, f, indent=2, sort_keys=True)
            
        print(f'Done. {len(scenes_downloaded)} complete scenes and {len(scenes_failed)} incomplete scenes downloaded')

    def try_downloading_view(self, url, path):
        for _ in range(3):
            try:
                urlretrieve(url, path)
                return True
            except:
                pass
            
        return False


# async def main():
#     d = PanopticScene('160224_ultimatum1')
#     await d.load()
#     d.cameras['vga'][0]

#     d = PanopticDownloader()
#     await d.load(scene_names_file='panoptic_scene_names.txt')

# main()
