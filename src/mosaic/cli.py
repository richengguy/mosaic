import click
import matplotlib.image as mpimg

from mosaic.imagelibrary import CIFAR100Library, ImageLibrary
from mosaic.index import Index
from mosaic.processing import (MosaicGenerator, assemble_mosiac,
                               assemble_source_grid)


@click.group()
@click.pass_context
def main(ctx: click.Context):
    ctx.ensure_object(CIFAR100Library)


@main.command()
@click.option('-d', '--dimensionality', nargs=1, default=128,
              help='Number of dimensions in the database feature space.')
@click.option('-l', '--label', 'labels', nargs=1, multiple=True,
              help='Specific labels to build the indices for.')
@click.pass_obj
def build_index(library: ImageLibrary, dimensionality: int, labels):
    '''Build the database indices needed for the photomosaic.

    The photomosaic needs to perform multiple look ups to find the best
    matching image for any image patch.  This will generate the indices for the
    given library.
    '''
    index = Index(ndim=dimensionality)
    if len(labels) == 0:
        click.secho('Warning: ', fg='yellow', bold=True, nl=False)
        click.echo('Building a complete index; this may take a while.')
        labels = None
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
@click.option('-g', '--source-grid', 'show_grid', is_flag=True,
              help='Generate a grid showing all unique source images alongside'
                   ' the mosaic.')
@click.argument('label', nargs=1)
@click.argument('image', nargs=1,
                type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.pass_obj
def generate(library: ImageLibrary, show_grid: bool, label, image):
    '''Generate a photomosaic for the given image.'''
    original = mpimg.imread(image)

    # Find tiles
    try:
        indexer, descriptors = Index.load(library, label)
        mosaic = MosaicGenerator(descriptors, indexer, (32, 32))
        tiles = mosaic.generate(original)
    except RuntimeError as e:
        raise click.ClickException(str(e)) from e

    # Generate output
    output = assemble_mosiac(tiles, library[label])

    if show_grid:
        image_grid = assemble_source_grid(tiles, library[label], 16, 3.25)
        mpimg.imsave(f'{image}-source.png', image_grid)

    mpimg.imsave(f'{image}-mosaic.png', output)


if __name__ == '__main__':
    main()
