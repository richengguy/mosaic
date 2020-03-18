from abc import ABC, abstractmethod, abstractproperty  # noqa: F401
from collections.abc import Mapping
import enum
import hashlib
import pathlib
import pickle
import shutil
from typing import FrozenSet, Iterator, List, Optional, Tuple, Union

import click
import numpy as np  # type: ignore
import requests
import tqdm  # type: ignore


@enum.unique
class HashType(enum.Enum):
    MD5 = enum.auto()
    SHA256 = enum.auto()


_HASHES = {
    HashType.MD5: hashlib.md5,
    HashType.SHA256: hashlib.sha256
}


PathLike = Union[str, pathlib.Path]


class ImageLibrary(ABC, Mapping):
    '''A library of images that are used to generate the photomosaics.

    The :class:`ImageLibrary` provides an API to access images within a
    collection.  These collections are, mostly, machine learning data sets.
    Subclasses will determine how to access any particular library.

    The library is implemented as a sequence on the internal library labels.
    This means that, for example, the ``[]`` operator access images in batches
    and not one at a time.  Singular access can be obtained using the various
    ``get_*()`` methods.

    Attributes
    ----------
    labels : FrozenSet[str], read-only
        all of the labels within the library
    '''
    def __init__(self, ident: str,
                 folder: PathLike = './libraries'):
        '''
        Parameters
        ----------
        ident : str
            a string identifier for the library
        folder : pathlib.Path
            top-level folder where the image libraries are stored
        '''
        folder = pathlib.Path(folder)
        self._libpath = (folder / ident).resolve()  # type: pathlib.Path
        self._first_init = False

        # Check if the directory exists, if not, create it.
        if not self._libpath.exists():
            self._first_init = True
            self._libpath.mkdir(parents=True, exist_ok=True)

    def __contains__(self, label: object) -> bool:
        return label in self.labels

    def __getitem__(self, label: str) -> List[np.ndarray]:
        return [
            self.get_image(index) for index in self.get_indices_for_label(label)  # noqa: E501
        ]

    def __iter__(self) -> Iterator[List[np.ndarray]]:
        for label in self.labels:
            yield self[label]

    def __len__(self) -> int:
        return len(self.labels)

    @abstractmethod
    def get_image(self, index: int) -> np.ndarray:
        '''Get an image from the library.

        This must be implemented by a subclass.

        Parameters
        ----------
        index : int
            the image's numerical index

        Returns
        -------
        np.ndarray
            the returned image
        '''

    @abstractmethod
    def get_indices_for_label(self, label: str) -> List[int]:
        '''Obtains the image indices for the given label.

        Parameters
        ----------
        label : str
            one of the labels registered with the library

        Returns
        -------
        List[int]
            list of image indices associated with that label
        '''

    @abstractmethod
    def number_of_images(self) -> int:
        '''Returns the number of images in the library.'''

    @abstractproperty
    def labels(self) -> FrozenSet[str]:
        '''The set of labels available within the image library.'''

    def _download(self, url: str, filename: str,
                  hash: Optional[Tuple[HashType, str]] = None) -> pathlib.Path:
        '''Download the contents of the URL to image library folder.

        The implementation is based on the one in
        https://sumit-ghosh.com/articles/python-download-progress-bar/

        Parameters
        ----------
        url : str
            download URL
        filename : str
            name of the downloaded file
        hash : (:class:`HashType`, hash)
            a tuple containing the hash type and the string used for the
            comparison

        Returns
        -------
        pathlib.Path
            path to the downloaded file
        '''
        green_checkmark = click.style('\u2713', fg='green', bold=True)
        red_cross = click.style('\u2717', fg='red', bold=True)

        path = self._libpath / filename

        if path.exists():
            return path

        click.secho('Downloading: ', bold=True, nl=False)
        click.echo(url)

        with path.open(mode='wb') as f:
            response = requests.get(url, stream=True)
            content_length = response.headers.get('content-length')

            if content_length is None:
                f.write(response.content)
            else:
                total_size = int(content_length)
                chunk_size = max(int(total_size/1000), 1024*1024)
                t = tqdm.tqdm(total=total_size, unit='B', unit_scale=True)
                for data in response.iter_content(chunk_size=chunk_size):
                    f.write(data)
                    t.update(len(data))
                t.close()

        if hash is not None:
            block_size = 65536
            hasher = _HASHES[hash[0]]()
            with path.open('rb') as f:
                data = f.read(block_size)
                while len(data) > 0:
                    hasher.update(data)
                    data = f.read(block_size)

            if hasher.hexdigest() == hash[1]:
                click.echo(f'Hashes match...{green_checkmark}')
            else:
                click.echo(f'Hashes don\'t match...{red_cross}')
                raise RuntimeError(f'Expected hash {hash[0]}, got {hasher.hexdigest()}')  # noqa: E501

        click.echo('Done...' + click.style('\u2713', fg='green'))
        click.secho('Saved to: ', bold=True, nl=False)
        click.echo(filename)
        return path


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
        self._names = self._labels['coarse_label_names']  # type: List[str]
        self._label_set = frozenset(self._names)
        self._ids = {i: name for i, name in enumerate(self._labels['coarse_label_names'])}  # noqa: E501

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

        target = self._names.index(label, 0, -1)
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
