import pathlib
import pickle
import shutil
from typing import FrozenSet, List

import click
import numpy as np

from .library import HashType, ImageLibrary
from .._types import PathLike


class CIFAR100Library(ImageLibrary):
    '''An image library composed of images from the CIFAR-100 dataset.'''
    def __init__(self, folder: PathLike = './libraries'):
        '''
        Parameters
        ----------
        folder : pathlib.Path, optional
            path to root library storage folder, by default './libraries'
        '''
        super().__init__('cifar100', folder=folder)
        tarball = self._download(
            'http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz',
            'cifar-100-python.tar.gz',
            (HashType.MD5, 'eb9058c3a382ffc7106e4002c42a8d85'))
        unpacked = self._unpack(tarball)
        self._load_images(unpacked)
        self._names = self._labels['coarse_label_names']
        self._label_set = frozenset(self._names)
        self._ids = {
            i: name
            for i, name in enumerate(self._labels['coarse_label_names'])
        }

    def number_of_images(self):
        return len(self._images['data'])

    def get_image(self, ind: int) -> np.array:  # type: ignore
        row = self._images['data'][ind, :]
        red = np.reshape(row[0:1024], (32, 32))
        green = np.reshape(row[1024:2048], (32, 32))
        blue = np.reshape(row[2048:3072], (32, 32))
        return np.dstack((red, green, blue))

    def get_indices_for_label(self, label: str) -> List[int]:
        if label not in self:
            raise KeyError(f'Unknown label {label}.')

        target = self._names.index(label, 0, len(self._names))
        return [
            image for image, label in enumerate(self._images['coarse_labels'])
            if label == target
        ]

    @property
    def labels(self) -> FrozenSet[str]:
        return self._label_set

    def _unpack(self, archive: pathlib.Path) -> pathlib.Path:
        '''Unpack the archive file at the given path.

        It will be extracted to the library's working directory.

        Parameters
        ----------
        archive : pathlib.Path
            path to the archive file

        Returns
        -------
        pathlib.Path
            path to the extracted archive
        '''
        path = archive.parent / 'cifar-100-python'
        if path.exists():
            return path

        click.secho('Extracting: ', bold=True, nl=False)
        click.echo(archive.name)

        shutil.unpack_archive(archive, self._libpath)

        if not path.exists():
            raise RuntimeError(f'Failed to unpack {archive}.')

        click.echo('Done...' + click.style('\u2713', fg='green', bold=True))
        return path

    def _load_images(self, unpacked: pathlib.Path):
        '''Load the images from the CIFAR-100 files.

        The files are standard Python pickle files.  Because they're relatively
        small (~150 MB), this just loads everything into memory.  There isn't
        really a point to store them on disk.

        Parameters
        ----------
        unpacked : pathlib.Path
            path to the unpacked archive
        '''
        meta = unpacked / 'meta'
        train = unpacked / 'train'

        with meta.open('rb') as f:
            self._labels = pickle.load(f, encoding='latin1')

        with train.open('rb') as f:
            self._images = pickle.load(f, encoding='latin1')
