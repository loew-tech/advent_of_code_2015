from typing import Iterable


def print_grid(grid: Iterable[Iterable], spacer='') -> None:
    for row in grid:
        print(spacer.join(str(i) for i in row))
