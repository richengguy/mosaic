import click
import matplotlib.image as mpimg  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

from mosaic.imagelibrary import ImageLibrary, CIFAR100Library
from mosaic.index import Index
from mosaic.processing import MosaicGenerator, assemble_image


@click.group()
@click.pass_context
def main(ctx: click.Context):
    ctx.ensure_object(CIFAR100Library)


@main.command()
@click.option('-l', '--label', 'labels', nargs=1, multiple=True,
              help='Specific labels to build the indices for.')
@click.pass_obj
def build_index(library: ImageLibrary, labels):
    '''Build the database indices needed for the photomosaic.

    The photomosaic needs to perform multiple look ups to find the best
    matching image for any image patch.  This will generate the indices for the
    given library.
    '''
    index = Index()
    index.build(library, labels)
    index.save(library.library_path)


@main.command()
@click.pass_obj
def labels(library: ImageLibrary):
    '''List the labels within the loaded library.'''
    click.secho('Available Labels:', bold=True)
    for label in sorted(list(library.labels)):
        click.echo(f'  {label}')


@main.command()
@click.argument('label', nargs=1)
@click.argument('image', nargs=1,
                type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.pass_obj
def generate(library: ImageLibrary, label, image):
    '''Generate a photomosaic for the given image.'''
    original = mpimg.imread(image)

    # Find tiles
    indexer, descriptors = Index.load(library, label)
    mosaic = MosaicGenerator(descriptors, indexer, (32, 32))
    tiles = mosaic.generate(original)

    # Generate output
    output = assemble_image(tiles, library[label])

    plt.imshow(tiles)

    plt.figure()
    plt.imshow(output)

    plt.show()

    mpimg.imsave(f'{image}-mosaic.jpg', output)


if __name__ == '__main__':
    main()
