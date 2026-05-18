import fsspec
import tarfile
import shutil
import os
import random
import requests


def download_from_tar(url, output_dir):
    url = "https://ml-site.cdn-apple.com/datasets/ca1m/train/ca1m-train-42444499.tar"
    output_dir = "downloaded_samples"
    fname = url.split('/')[-1]
    fbase = os.path.splitext(fname)[0].split('-')[-1]

    os.makedirs(output_dir, exist_ok=True)

    with fsspec.open(url) as f:
        with tarfile.open(fileobj=f, mode='r') as tar:
            # Get the list of the first 100 members
            members = tar.getmembers()
            members = [(i, m) for i, m in enumerate(members)
                       if '.gt/depth.png' in m.get_info()['name']]
            random.shuffle(members)
            # gets 10% of depth maps
            members = members[:int(len(members) * 0.1)]
            size = sum([m.size for _, m in members])

            print(f'\nDownloading `{url}` ({size / (1024 * 1024)} MiB)')

            for i, m in members:
                name = f'{fbase}_{i:05}_depth.png'
                target_path = os.path.join(output_dir, name)

                print(f"  Fetching: `{name}`")

                with tar.extractfile(m) as source:
                    with open(target_path, "wb") as target:
                        shutil.copyfileobj(source, target)

    print(f'Ended downloading `{url}`')


def download_from_files(txt_url, output_dir):
    response = requests.get(txt_url)

    if response.status_code == 200:
        content = response.text

        urls = content.strip().splitlines()
    else:
        raise Exception(f'Error {response.status_code}')

    for url in urls:
        download_from_tar(url, output_dir)


if __name__ == '__main__':
    txt_url = 'https://raw.githubusercontent.com/apple/ml-cubifyanything/refs/heads/main/data/train.txt'
    output_dir = 'dataset_apple_ml_depths'
    download_from_files(txt_url, output_dir)
