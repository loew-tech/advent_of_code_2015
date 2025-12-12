import os.path
from http import HTTPStatus
import requests
from typing import List, Callable

from constants import ADVENT_URI, INPUTS_PATH


def read_input(
        day: int | str,
        year: int | str = 2015,
        delim: str | None = '\n',
        parse: Callable[[str], any] = None
) -> List[any] | str:
    if os.path.exists(f'{INPUTS_PATH}{day}.txt'):
        with open(f'{INPUTS_PATH}{day}.txt') as in_:
            return _process_input(in_.read(), delim, parse)

    with open('.env') as env_:
        session_id = env_.read().strip().split('\n')[0]
    response = requests.get(f'{ADVENT_URI}{year}/day/{day}/input',
                            cookies={'session': session_id})

    if not response.status_code == HTTPStatus.OK:
        raise Exception(f'Failed to acquire input from {ADVENT_URI}')

    if not os.path.exists(INPUTS_PATH):
        os.mkdir(INPUTS_PATH)
    with open(f'{INPUTS_PATH}{day}.txt', 'w') as out:
        out.write(response.text)
    return _process_input(response.text, delim, parse)


def _process_input(
        text, delim: str,
        parse: Callable[[List[str] | str], any]
) -> any:
    data = text.strip().split(delim) if delim else text
    return data if parse is None else [
        parse(e) for e in (data if type(data) == list else [data])
    ]


def get_inbounds(
        grid: List[List[any] | str]
) -> Callable[[int, int], bool]:
    return lambda y, x: inbounds(y, x, grid)


def inbounds(y, x: int, grid: List[List[any] | str]) -> bool:
    return 0 <= y < len(grid) and 0 <= x < len(grid[y])
