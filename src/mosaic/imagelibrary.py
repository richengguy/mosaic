import abc
import enum
import hashlib
import pathlib
import pickle
import secrets
import shutil
from typing import Optional, Dict, Tuple

import click
import requests
import tqdm


@enum.unique
class HashType(enum.Enum):
    MD5 = enum.auto()
    SHA256 = enum.auto()


_HASHES = {
    HashType.MD5: hashlib.md5,
    HashType.SHA256: hashlib.sha256
}


class ImageLibrary(abc.ABC):
    '''A library of images that are used to generate the photomosaics.

    The :class:`ImageLibrary` provides an API to access images within a
    collection.  These collections are, mostly, machine learning data sets.
    '''
    def __init__(self, ident: str,
                 folder: pathlib.Path = './libraries'):
        '''
        Parameters
        ----------
        ident : str
            a string identifier for the library
        folder : pathlib.Path
            top-level folder where the image libraries are stored
        '''
        self._libpath = (folder / ident).resolve()  # type: pathlib.Path

        # Check if the directory exists, if not, create it.
        if not self._libpath.exists():
            self._libpath.mkdir(parents=True, exist_ok=True)

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
        click.secho('Downloading: ', bold=True, nl=False)
        click.echo(url)

        if path.exists():
            click.echo('File exists...nothing to do.')
            return path

        with path.open(mode='wb') as f:
            response = requests.get(url, stream=True)
            total_size = response.headers.get('content-length')

            if total_size is None:
                f.write(response.content)
            else:
                total_size = int(total_size)
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
    def __init__(self, folder=pathlib.Path('./libraries')):
        super().__init__('cifar100', folder=folder)
        tarball = self._download(
            'http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz',
            'cifar-100-python.tar.gz',
            (HashType.MD5, 'eb9058c3a382ffc7106e4002c42a8d85'))
        unpacked = self._unpack(tarball)
        self._load_images(unpacked)

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
        click.secho('Extracting: ', bold=True, nl=False)
        click.echo(archive.name)

        path = archive.parent / 'cifar-100-python'
        if path.exists():
            click.echo('Folder exists...nothing to do.')
            return path

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
