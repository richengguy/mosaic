from typing import Dict, List

import click
import hnswlib
import numpy as np

from mosaic.imagelibrary import ImageLibrary
from mosaic.processing import FeatureGenerator

_GREEN_CHECKMARK = click.style('\u2713', fg='green', bold=True)
_RED_CROSS = click.style('\u2717', fg='red', bold=True)


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
    '''
    def __init__(self, images: List[np.ndarray], dimensionality: int = 128):
        click.echo(f' - Generating image features', nl=False)
        self.descriptors = FeatureGenerator(images, dimensionality)
        click.echo(f'...{_GREEN_CHECKMARK}')

        click.echo(f' - Building search index', nl=False)
        self.indexer = hnswlib.Index('l2', dimensionality)
        self.indexer.init_index(max_elements=len(images))
        self.indexer.add_items(self.descriptors.descriptors)
        click.echo(f'...{_GREEN_CHECKMARK}')


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
    def __init__(self, ndim: int = 128):
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

    def build(self, library: ImageLibrary):
        '''Build the index from the given library.

        One index is built for each category/class within the library.

        Parameters
        ----------
        library : ImageLibrary
            the input image library
        '''
        click.secho('Building Library Index', bold=True)
        click.echo('----')
        for label in library.labels:
            click.secho(f'{label}:', bold=True)
            self._indices[label] = Category(library[label])
