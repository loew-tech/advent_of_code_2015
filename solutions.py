import inspect
import sys

from dbg_utils import get_expected

if __name__ == '__main__':
    def print_result(d: int,
                     expected, part: str,
                     result: int | str) -> None:
        if expected == str(result):
            print(f'PASS {d}{part}: {d}({part}) = {result}.')
            return
        print(f'FAILED {d}{part}: Expected = {expected}. {d}({part}) ='
              f' {result}')

    def test_days(days):
        for day_ in days:
            if day_ not in funcs:
                print(f'{day}()= NotImplemented')
                continue
            expected, result = get_expected(day=day_), funcs[day_](test=True)
            print_result(day_, expected, 'A', result)

            expected = get_expected(day=day_, part='B')
            result = funcs[day_](test=True, part='B')
            print_result(day_, expected, 'B', result)


    testing = '-t' in sys.argv[1:]
    args = (f'day_{i}' for i in (sys.argv[1:] if
            sys.argv[1:] else range(1, 26)) if type(i) == int or i.isnumeric())
    members = inspect.getmembers(inspect.getmodule(inspect.currentframe()))
    funcs = {name: member for name, member in members
             if inspect.isfunction(member)}

    if testing:
        test_days(args)

    for day in args:
        if day not in funcs:
            print(f'{day}() = NotImplemented')
            continue
        print(f'{day}() = {funcs[day]()}')
        print(f'{day}(part="B") = {funcs[day](part="B")}')