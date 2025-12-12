import os
from typing import Iterable

from bs4 import BeautifulSoup
import requests

from constants import ADVENT_URI, TESTS_PATH


def get_test_input(
        day: int | str,
        year: int | str = 2015,
) -> str:
    if os.path.exists(f'{TESTS_PATH}{day}.txt'):
        with open(f'{TESTS_PATH}{day}.txt') as in_:
            return in_.read()

    problem = requests.get(f'{ADVENT_URI}{year}/day/{day}')
    soup = BeautifulSoup(problem.content, 'html.parser')
    input_ = soup.find('pre').code
    txt = input_.get_text()
    if not os.path.exists(TESTS_PATH):
        os.mkdir(TESTS_PATH)
    with open(f'{TESTS_PATH}{day}.txt', 'w') as out:
        out.write(txt)
    return txt


def get_expected(
        day: int | str,
        year: int | str = 2025,
        part='A'
) -> str:
    if not os.path.exists(TESTS_PATH):
        os.mkdir(TESTS_PATH)
    file_ = f'{TESTS_PATH}{day}{part}_expected.txt'
    if os.path.exists(file_):
        with open(file_) as in_:
            return in_.read()

    cookies = None
    if not part.upper() == 'A':
        with open('.env') as env_:
            session_id = env_.read().strip().split('\n')[0]
            cookies = {'session': session_id}
    problem = requests.get(f'{ADVENT_URI}{year}/day/{day.replace("day_", "")}',
                           cookies=cookies)
    soup = BeautifulSoup(problem.content, 'html.parser')

    if not part.upper() == 'A' and 'Part Two' not in soup.prettify():
        return 'UNLOCK PART 2 TO TEST'
    expected = ''
    for p in soup.find_all('p'):
        for c in p.find_all('code'):
            if em := c.find('em'):
                expected = em.get_text()

    if not expected.strip():
        for p in soup.find_all('p'):
            for em in p.find_all('em'):
                if code := em.find('code'):
                    expected = code.get_text()

    with open(file_, 'w') as out:
        out.write(expected)
    return expected


def print_grid(grid: Iterable[Iterable], spacer='') -> None:
    for row in grid:
        print(spacer.join(str(i) for i in row))
