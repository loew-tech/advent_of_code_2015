from hashlib import md5
import inspect
import sys
from typing import List

from classes import Box
from constants import CARDINAL_DIRECTIONS
from utils import read_input


def day_1(part_1=True) -> int:
    data = read_input(day=1, delim=None)
    if part_1:
        return data.count('(') - data.count(')')

    floor = 0
    for i, inc in enumerate(data, start=1):
        floor += 1 if inc == '(' else -1
        if floor == -1:
            return i


def day_2(part_1=True) -> int:
    boxes: List[Box] = read_input(day=2,
                                  parse=lambda ln: Box(*map(int,
                                                            ln.split('x')))
                                  )
    if part_1:
        return sum(box.wrapping_paper for box in boxes)
    return sum(box.ribbon for box in boxes)


def day_3(part_1=True) -> int:
    directions, visited = read_input(day=3, delim=None), {(0, 0)}
    incs = dict(zip('v<>^', CARDINAL_DIRECTIONS))
    y_santa = x_santa = y_bot = x_bot = 0
    for i, dir_ in enumerate(directions):
        yi, xi = incs[dir_]
        if part_1 or not i % 2:
            y_santa += yi
            x_santa += xi
            visited.add((y_santa, x_santa))
            continue
        y_bot += yi
        x_bot += xi
        visited.add((y_bot, x_bot))
    return len(visited)


def day_4(part_1=True) -> int:
    key, cnt, num_zeroes = read_input(day=4, delim=None), 0, 5 + (not part_1)
    k, zeroes = md5((key + str(cnt)).encode()).hexdigest(), '0' * num_zeroes
    while not k[:num_zeroes] == zeroes and (cnt := cnt+1):
        k = md5(f'{key}{cnt}'.encode()).hexdigest()
    return cnt


def day_5(part_1=True) -> int:
    strings, cnt = read_input(day=5), 0
    bad, vowels = {'ab', 'cd', 'pq', 'xy'}, {'a', 'e', 'i', 'o', 'u'}

    for s in strings:
        prev, bad_, double_letter, vowel_cnt = None, False, False, 0
        for c in s:
            bad_ |= f'{prev}{c}' in bad
            double_letter |= prev == c
            vowel_cnt += c in vowels
            prev = c
        cnt += not bad_ and double_letter and vowel_cnt >= 3
    return cnt


if __name__ == '__main__':
    args = (f'day_{i}' for i in (sys.argv[1:] if
                                 sys.argv[1:] else range(1, 26)) if
            type(i) == int or i.isnumeric())
    members = inspect.getmembers(inspect.getmodule(inspect.currentframe()))
    funcs = {name: member for name, member in members
             if inspect.isfunction(member)}
    for day in args:
        if day not in funcs:
            print(f'{day}() = NotImplemented')
            continue
        print(f'{day}() = {funcs[day]()}')
        print(f'{day}(part="B") = {funcs[day](part_1=False)}')
