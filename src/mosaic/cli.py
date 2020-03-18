import click

from mosaic.imagelibrary import ImageLibrary, CIFAR100Library


@click.group()
@click.pass_context
def main(ctx: click.Context):
    ctx.ensure_object(CIFAR100Library)


@main.command()
@click.pass_obj
def labels(library: ImageLibrary):
    '''List the labels within the loaded library.'''
    click.secho('Available Labels:', bold=True)
    for label in sorted(list(library.labels)):
        click.echo(f'  {label}')


if __name__ == '__main__':
    main()
