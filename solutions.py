import inspect
import sys

from classes import Box
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
    def surface_area(b: Box) -> int:
        return 2 * b.l * b.w + 2 * b.w * b.h + 2 * b.h * b.l

    def total_wrapping_paper(box_: Box):
        sorted_ = sorted(box_)
        return surface_area(box_) + sorted_[0] * sorted_[1]

    boxes = read_input(day=2, parse=lambda ln: Box(*map(int, ln.split('x'))))
    return sum(total_wrapping_paper(box) for box in boxes)


if __name__ == '__main__':
    args = (f'day_{i}' for i in (sys.argv[1:] if
            sys.argv[1:] else range(1, 26)) if type(i) == int or i.isnumeric())
    members = inspect.getmembers(inspect.getmodule(inspect.currentframe()))
    funcs = {name: member for name, member in members
             if inspect.isfunction(member)}
    for day in args:
        if day not in funcs:
            print(f'{day}() = NotImplemented')
            continue
        print(f'{day}() = {funcs[day]()}')
        print(f'{day}(part="B") = {funcs[day](part_1=False)}')
