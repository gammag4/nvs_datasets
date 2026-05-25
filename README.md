# NVS Datasets

This repo contains scripts to download and process Novel View Synthesis datasets.

## WildRGBD

To download the WildRGBD dataset, simply run:

```sh
python download_wildrgbd.py --cat all --path "dataset/destination/path"
```

Or to download a single category, run this:

```sh
python download_wildrgbd.py --cat "category" --path "dataset/destination/path"
```

You need the packages `wget`, `zip` and `unzip` installed.

## License

Unless otherwise noted, files in this repository are licensed under the MIT License. See `LICENSE`.

Some scripts were copied or adapted from third-party repositories and retain their original licenses.

Per-file license information is included at the top of each file.
