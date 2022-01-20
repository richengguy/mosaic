import pathlib
from typing import Dict, List, Optional, Union

import click
import hnswlib  # type: ignore
import numpy as np

from mosaic.imagelibrary import ImageLibrary
from mosaic.processing import FeatureGenerator

_GREEN_CHECKMARK = click.style('\u2713', fg='green', bold=True)
_RED_CROSS = click.style('\u2717', fg='red', bold=True)
PathLike = Union[str, pathlib.Path]


class Category(object):
    '''A single category within the main index.

    A "category" is just the collection of images that is used to construct a
    single library index.  It contains the kNN indexing structure and the
    mechanism to generate feature descriptors.

    Attributes
    ----------
    indexer : hnswlib.Index
        the kNN indexing data structure
    descriptors : FeatureGenerator
        object used for generating feature descriptors for the image collection
    tile_size : (height, width)
        image/tile size of images in this category
    '''
    def __init__(self, images: List[np.ndarray], dimensionality: int = 256):
        click.echo(' - Generating image features', nl=False)
        self.descriptors = FeatureGenerator(images, dimensionality)
        click.echo(f'...{_GREEN_CHECKMARK}')

        click.echo(' - Building search index', nl=False)
        self.indexer = hnswlib.Index('l2', dimensionality)
        self.indexer.init_index(max_elements=len(images))
        self.indexer.add_items(self.descriptors.descriptors)
        click.echo(f'...{_GREEN_CHECKMARK}')

        self.tile_size = images[0].shape[0:2]


class Index(object):
    '''Indexes images in an image library for easy retrieval.

    The :class:`Index` is a wrapper around hnswlib to make working with it
    easier.  It also performs some of the necessary pre-processing to use the
    fast nearest-neighbour library.

    Attributes
    ----------
    initialized : bool
        if ``False`` then the index has not been built yet
    '''
    def __init__(self, ndim: int = 256):
        '''
        Parameters
        ----------
        ndim : int, optional
            dimensionality of the search space; defaults to '128'
        '''
        self._ndim = ndim
        self._indices: Dict[str, Category] = {}

    @property
    def initialized(self) -> bool:
        return len(self._indices) != 0

    def build(self, library: ImageLibrary, labels: Optional[List[str]] = None):
        '''Build the index from the given library.

        One index is built for each category/class within the library.

        Parameters
        ----------
        library : ImageLibrary
            the input image library
        labels : List[str], optional
            if provided, only generate indices for these labels
        '''
        click.secho('Building Library Index', bold=True)
        click.echo('----')
        click.echo(click.style('Feature Size: ', bold=True) + f'{self._ndim}')

        if labels is None:
            categories = library.labels
        else:
            categories = frozenset(labels)

        for label in categories:
            click.secho(f'{label}:', bold=True)
            self._indices[label] = Category(library[label])

    def save(self, folder: PathLike):
        '''Save the index to disk.

        Parameters
        ----------
        folder : PathLike
            folder to where the indices should be stored
        '''
        folder = pathlib.Path(folder)
        for label, category in self._indices.items():
            index_file = folder / f'{label}.index'
            descr_file = folder / f'{label}.descriptors'
            category.indexer.save_index(index_file.as_posix())
            category.descriptors.save(descr_file)

    @staticmethod
    def load(library: ImageLibrary, label: str):
        index_file = library._libpath / f'{label}.index'
        descr_file = library._libpath / f'{label}.descriptors'

        if not index_file.exists():
            raise RuntimeError(f"Could not find an index for '{label}' label.")

        indexer = hnswlib.Index('l2', 256)
        indexer.load_index(index_file.as_posix())
        descriptors = FeatureGenerator.load(descr_file)

        return indexer, descriptors
