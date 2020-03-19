from typing import List, Tuple

import numpy as np


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

        return x @ self.principal_components

    @property
    def dimensionality(self) -> int:
        return self._dimensionality

    @property
    def input_features(self) -> int:
        return self._input_features
