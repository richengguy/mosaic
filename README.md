# mosaic
"mosaic" is a small Python library/app to generate photographic mosaics.  This
is where you take an original image and turn it into a composition of image
tiles.  For example,


| Original | Mosaic (Click to see full-sized version) |
| -------- | ------- |
| ![original-thumb] | [![mosaic-thumb]][mosaic-full] |

[original-thumb]: docs/branch-thumb.jpg "Original Thumbnail"
[mosaic-thumb]: docs/branch-mosaic-thumb.jpg "Mosaic Thumbnail"
[mosaic-full]: docs/branch-mosaic.jpg

It was originally created as part of a photography course project.  Aside from
generating/rendering the mosaic, it doesn't really have that many features. In
fact, the way it's currently built will only let you generate mosaics from the
[CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.  It's
extendable, but I (the author) don't have any plans to update it.

## Installation

The only real requirement for "mosaic" is to make sure you have
[conda](https://docs.conda.io/en/latest/) available on your system.  The
environment and dependencies are specified in [environment.yml](environment.yml)
and [requirements.txt](requirements.txt), respectively.

To setup the conda/Python environment, just run:

```bash
$ conda env create
```

Once done, use

```
$ conda activate mosaic
```

to activate it.

You can then use the `mosaic` command generate the feature database and render
mosaics.

## Usage

`mosaic` will automatically download the CIFAR-100 dataset the first
time you run it.  It will store it in a "libraries" folder.  You can get a full
list of commands with `mosaic --help`.

### Building Feature Database

The feature database index needs to be built before the mosaics are generated.
You can generate it for all labels or for just one.  For example, to build an
index for trees:

```
$ mosaic build-index -l trees
```

The index is cached in the "libraries" folder and only needs to be generated
once.

### Rendering Photographic Mosiacs

Rendering mosaics is done with `mosaic generate`.  For example, to render a
mosaic using the "trees" label:

```
mosaic generate trees my-image.jpg
```

This will generate a "my-image-mosaic.png" file in the *same folder as the
source image*.

## How it works

The mosaic generation algorithm is pretty simple.  It has two parts,

1. Feature Database Generation
2. Mosaic Rendering

Generating the feature database is nothing more than a PCA over the image
library you wish to use for the mosaic.  Each image is projected onto a
D-dimensional (D=128 is the default) subspace and the resulting vectors are
indexed using [hsnwlib](https://github.com/nmslib/hnswlib).

Once you have a source image, the mosaic generation algoirhtm is:

1. Split the source image up into NxN tiles (currently uses 32x32).
2. For each tile:
    1. Compute a feature descriptor by projecting the tile using the principle
       components found when creating the feature database.
    2. Perform a fast-approximate nearest neighbour search to find the
       best-matching image from the image library.
    3. Put the *index* of the best matching tile into an best-matching index
       array.
3. For each element in the index array, blit the tile into the output mosaic.
4. (Optional) For each unique entry in the index array, blit the tile into a
   "tile selection" image.

## Why CIFAR?

The project was to highlight how labelling decisions in ML systems are, at the
end of the day, choices made by people.  Ignoring unsupervised learning (not in
scope), the decision of what consitutes a "tree" superclass in the CIFAR-100
dataset is somewhat arbitrary.  This is meant to be visually represented by the
mosaic.  So, in the example above, it's a picture of a tree branch that's
comprised of tiles of what someone *else* thought was tree.

I chose CIFAR-100 because it's a relatively small dataset.  I didn't need a
large and comprehensive ML dataset for the course project.  CIFAR-100 was large
enough to have some variety in the images without needing to download gigabytes
of data.  Basically it was enough for what I needed.
