import abc
import pathlib

import click
import requests
import tqdm


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

    def download(self, url: str, filename: str):
        '''Download the contents of the URL to image library folder.

        The implementation is based on the one in
        https://sumit-ghosh.com/articles/python-download-progress-bar/

        Parameters
        ----------
        url : str
            download URL
        filename : str
            name of the downloaded file
        '''
        path = self._libpath / filename
        click.secho('Downloading: ', bold=True, nl=False)
        click.echo(url)

        if path.exists():
            click.echo('File exists...nothing to do.')
            return

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

        click.echo('Done...' + click.style('\u2713', fg='green'))
        click.secho('Saved to: ', bold=True, nl=False)
        click.echo(filename)


class CIFAR100Library(ImageLibrary):
    '''An image library composed of images from the CIFAR-100 dataset.'''
    def __init__(self, folder=pathlib.Path('./libraries')):
        super().__init__('cifar100', folder=folder)
        self.download(
            'http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz',
            'cifar-100-python.tar.gz')
