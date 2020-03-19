import click

from mosaic.imagelibrary import ImageLibrary, CIFAR100Library
from mosaic.index import Index


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


if __name__ == '__main__':
    main()
