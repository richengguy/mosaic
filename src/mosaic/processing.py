import pathlib
import pickle
from typing import List, Tuple, Iterator

import click
import hnswlib  # type: ignore
import numpy as np
import scipy.ndimage  # type: ignore
import tqdm  # type: ignore


def _svd(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''Performs an SVD on the provided array.

    Per the NumPy documentation (and any linear algebra text book), this
    factorizes ``X`` into a matrix such that ``X = U @ np.diag(S) @ Vt``.  This
    just wraps NumPy's :func:`svd` function and then applies scikit-learn's
    "svd flip" algorithm to ensure some consistency on the SVD results.

    Note
    ----
    The 'K' below is the minimum of 'M' and 'N'.

    Parameters
    ----------
    X : numpy.ndarray
        input MxN array

    Returns
    -------
    U : numpy.ndarray
        SVD's left unitary matrix; will be MxK
    S : numpy.ndarray
        SVD's singular values; will be of length K
    Vt : numpy.ndarray
        SVD's right unitary matrix; will be KxN
    '''
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    # Check the signs to ensure SVD consistency (see
    # https://stackoverflow.com/a/44847053), which is what scikit-learn's
    # svd_flip() function does; see
    # https://github.com/scikit-learn/scikit-learn/blob/0.22.2/sklearn/utils/extmath.py#L526
    max_inds = np.argmax(np.abs(U), axis=0)
    signs = np.sign(U[max_inds, range(U.shape[1])])
    U *= signs
    Vt *= signs[:, np.newaxis]

    return U, S, Vt


def assemble_mosiac(grid: np.ndarray, images: List[np.ndarray]) -> np.ndarray:
    '''Assemble a mosaic image from a grid and the set of images.

    Parameters
    ----------
    grid : np.ndarray
        a MxN array containing image IDs for each tile location
    images : List[np.ndarray]
        list of image tiles

    Returns
    -------
    np.ndarray
        a HxW image generated from the original input list
    '''
    tile_height, tile_width, _ = images[0].shape
    out_height = grid.shape[0]*tile_height
    out_width = grid.shape[1]*tile_width

    output = np.zeros((out_height, out_width, 3), dtype=np.uint8)

    progress = tqdm.tqdm(total=grid.shape[0]*grid.shape[1],
                         desc='Assembling mosaic', unit='tile')
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            r_start = r*tile_height
            c_start = c*tile_width

            r_end = r_start + tile_height
            c_end = c_start + tile_width

            output[r_start:r_end, c_start:c_end, :] = images[grid[r, c]]

            progress.update(1)

    progress.close()

    return output


def assemble_source_grid(grid: np.ndarray, images: List[np.ndarray],
                         margin: int, scale: float = 1.0) -> np.ndarray:
    '''Assemble an image grid showing the unique source images for a mosaic.

    The output image grid attempts to maintain the same size as the photomosaic
    when its generated via :func:`assemble_image`.

    Parameters
    ----------
    grid : np.ndarray
        an MxN array containing image IDs for each tile location
    images : List[np.ndarray]
        list of image tiles
    margin : int
        margin around each image
    scale : float, optional
        scale the tile images by some amount, defaults to '1' or no scaling

    Returns
    -------
    np.ndarray
        an image grid of all unique images in the photomosaic, sorted by ID
    '''
    # Compute the mosaic dimensions.
    mosiac_height = grid.shape[0]*images[0].shape[0]
    mosiac_width = grid.shape[1]*images[1].shape[1]

    # Compute the size of the output tiles.
    if scale > 1.0:
        shape = scipy.ndimage.zoom(images[0], (scale, scale, 1)).shape
    else:
        shape = images[0].shape

    tile_height, tile_width, _ = shape

    # Compute the image grid dimensions
    cell_height = tile_height + margin
    cell_width = tile_width + margin

    ids = np.unique(grid)
    N = len(ids)

    cols = mosiac_width // cell_width
    rows = int(np.ceil(N / cols))

    # Compute the output size and various offsets
    output_height = max(rows*cell_height, mosiac_height)
    output_width = max(cols*cell_width, mosiac_width)

    padding = int(margin / 2)
    centering_offset = (output_width - cols*cell_width) // 2

    # Render the grid.
    output = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    output[:] = 255

    progress = tqdm.tqdm(total=N, desc='Assembling source grid', unit='tile')
    for r in range(rows):
        for c in range(cols):
            i = c + r*cols
            if i == N:
                break

            tile = images[ids[i]]
            if scale > 1.0:
                tile = scipy.ndimage.zoom(tile, (scale, scale, 1))

            r_start = r*(tile_height + margin) + padding
            c_start = c*(tile_width + margin) + padding + centering_offset

            r_end = r_start + tile_height
            c_end = c_start + tile_width

            output[r_start:r_end, c_start:c_end, :] = tile

            progress.update(1)

    progress.close()

    return output


class FeatureGenerator(object):
    '''PCA-based feature descriptors for an image collection.

    Attributes
    ----------
    dimensionality : int
        number of dimensions in the descriptors
    input_features : int
        number of input features
    mean : numpy.ndarray
        the mean used for centering the data used to generate the descriptors
    stddev : numpy.ndarray
        the standard deviation used for standardizing the data used to generate
        the descriptors
    '''
    def __init__(self, images: List[np.ndarray], dimensionality: int = 128):
        '''
        Parameters
        ----------
        images : List[np.ndarray]
            list of images
        dimensionality : int
            number of dimensions in the descriptor space
        '''
        num_images = len(images)
        num_features = np.prod(images[0].shape)

        self._dimensionality = dimensionality
        self._input_features = num_features

        # Flatten the images into a single array of feature vectors.
        data = np.zeros((num_images, num_features))
        for i, image in enumerate(images):
            data[i, :] = image.flatten()

        # Normalize and standardize the image feature vectors.
        self.mean = np.mean(data, axis=0)
        self.stddev = np.std(data, axis=0)
        data = (data - self.mean) / self.stddev

        # Compute eigenvectors using an SVD.
        U, S, V = _svd(data)

        # Compute the principal components and descriptors.
        self.principal_components = V[:self._dimensionality]
        self.descriptors = U[:, :self._dimensionality]*S[:self._dimensionality]

    def compute(self, image: np.ndarray) -> np.ndarray:
        '''Compute a feature descriptor for the provided image.

        The descriptor is computed by using the principal components to project
        the higher-dimension image vector into the lower-dimension descriptor
        space.

        Parameters
        ----------
        image : np.ndarray
            input image

        Returns
        -------
        np.ndarray
            feature descriptor

        Raises
        ------
        ValueError
            if the unwrapped image isn't the same size as the images used to
            generate the descriptor space
        '''
        x = image.flatten()
        if x.size != self.input_features:
            raise ValueError(f'The image must have {self.input_features} elements.')  # noqa: E501

        # Scale to the same range as the "training" data.
        x = (x - self.mean) / self.stddev

        return x @ self.principal_components.T

    @property
    def dimensionality(self) -> int:
        return self._dimensionality

    @property
    def input_features(self) -> int:
        return self._input_features  # type: ignore

    def save(self, path: pathlib.Path):
        '''Save the descriptor structure to disk.

        Parameters
        ----------
        path : pathlib.Path
            path where the descriptor should be stored
        '''
        with path.open('wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: pathlib.Path):
        '''Load the descriptor structure from disk.

        Parameters
        ----------
        path : pathlib.Path
            path to the where the discriptors are stored
        '''
        with path.open('rb') as f:
            return pickle.load(f)


class MosaicGenerator(object):
    '''Generate photomosaics from image databases.'''

    class _Tiles(object):
        '''Used to generate the tiles for a particular image.'''
        def __init__(self, image: np.ndarray, tile_size: Tuple[int, int]):
            height, width, _ = image.shape

            self._image = image
            self._tilesize = tile_size

            self._tiles_x = width // tile_size[0]
            self._tiles_y = height // tile_size[1]

            self._output_width = self._tiles_x*tile_size[0]
            self._output_height = self._tiles_y*tile_size[1]

            delta_width = width - self._output_width
            delta_height = height - self._output_height

            self._x_start = delta_width // 2
            self._y_start = delta_height // 2

            self._x_end = (width - 1) - (delta_width // 2)
            self._y_end = (height - 1) - (delta_height // 2)

        def __len__(self) -> int:
            return self._tiles_x*self._tiles_y

        def __iter__(self) -> Iterator[Tuple[int, int, np.ndarray]]:
            height, width = self._tilesize
            for r in range(self._tiles_y):
                for c in range(self._tiles_x):
                    r_start = r*height
                    c_start = c*width

                    r_end = r_start + height
                    c_end = c_start + width

                    yield r, c, self._image[r_start:r_end, c_start:c_end, :]

        @property
        def grid_size(self) -> Tuple[int, int]:
            '''Size of the tile grid.

            Returns
            -------
            grid_rows : int
                number of grid rows
            grid_cols : int
                number of grid columns
            '''
            return self._tiles_y, self._tiles_x

        @property
        def output_size(self) -> Tuple[int, int, int]:
            '''Size of the final output image.

            Returns
            -------
            height, width, channels : int
                the image output size.
            '''
            return self._output_height, self._output_width, self._image.shape[2]  # noqa: E501

    def __init__(self, features: FeatureGenerator, indexer: hnswlib.Index,
                 tile_size: Tuple[int, int]):
        self._features = features
        self._indexer = indexer
        self._tile_size = tile_size

    def generate(self, image: np.ndarray) -> np.ndarray:
        '''Generate a mosaic for the given image.

        Parameters
        ----------
        image : np.ndarray
            input RGB image

        Returns
        -------
        np.ndarray
            a 2D array where each element is an ID into the image library; this
            can then be used to assemble the final mosiac
        '''
        tiles = MosaicGenerator._Tiles(image, self._tile_size)

        click.echo(click.style('Tile Size: ', bold=True) +
                   f'{self._tile_size[0]}x{self._tile_size[1]}')
        click.echo(click.style('Grid Size: ', bold=True) +
                   f'{tiles.grid_size[0]}x{tiles.grid_size[1]}')
        click.echo(click.style('Expected Output Size: ', bold=True) +
                   f'{tiles.output_size[0]}x{tiles.output_size[1]}')

        output = np.zeros(tiles.grid_size, dtype=int)
        for r, c, tile in tqdm.tqdm(tiles, unit='tile', desc='Finding tiles'):
            desc = self._features.compute(tile)
            index, _ = self._indexer.knn_query(desc)
            output[r, c] = np.squeeze(index)

        return output
