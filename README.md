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

It was originally created for a photography project so it doesn't have that many
features beyond specifying the image to generate a mosiac for.  In fact, the way
it's currently built will only let it generate mosaics from the
[CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html).  It's extendable, but
I (the author) don't have any plans to update it.
