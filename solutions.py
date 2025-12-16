import copy
import heapq
from collections import defaultdict
from hashlib import md5
import inspect
import sys
from typing import List, Dict

from classes import Box, LightInterval, LogicGate, Edge
from constants import CARDINAL_DIRECTIONS
from helpers import parse_day_7
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
    while not k[:num_zeroes] == zeroes and (cnt := cnt + 1):
        k = md5(f'{key}{cnt}'.encode()).hexdigest()
    return cnt


def day_5(part_1=True) -> int:
    strings, cnt = read_input(day=5), 0
    bad, vowels = {'ab', 'cd', 'pq', 'xy'}, {'a', 'e', 'i', 'o', 'u'}
    for s in strings:
        pp_prev, p_prev, prev = None, None, None
        bad_, double_letter, vowel_cnt = False, False, 0
        pair, spaced_pair = False, False
        pairs = set()
        for c in s:
            bad_ |= f'{prev}{c}' in bad
            double_letter |= prev == c
            vowel_cnt += c in vowels

            pair |= (p := f'{prev}{c}') in pairs and (
                    not p == f'{p_prev}{prev}' or
                    p == f'{pp_prev}{p_prev}')
            pairs.add(p)
            spaced_pair |= p_prev == c
            pp_prev, p_prev, prev = p_prev, prev, c
        cnt += part_1 and not bad_ and double_letter and vowel_cnt >= 3
        cnt += not part_1 and pair and spaced_pair
    return cnt


def day_6(part_1=True) -> int:
    def parse(ln: str) -> LightInterval:
        line = ln.split()
        action, i = (line[0], 1) if line[0] == 'toggle' else (line[1], 2)
        coord1, _, coord2 = line[i:]
        x0, y0 = map(int, coord1.split(','))
        x1, y1 = map(int, coord2.split(','))
        return LightInterval(action, x0, y0, x1, y1)

    def turn_on(intvrl: LightInterval) -> None:
        for y in range(intvrl.y0, intvrl.y1 + 1):
            for x in range(intvrl.x0, intvrl.x1 + 1):
                brightness[(y, x)] += 1

    def turn_off(intvrl: LightInterval) -> None:
        for y in range(intvrl.y0, intvrl.y1 + 1):
            for x in range(intvrl.x0, intvrl.x1 + 1):
                if part_1:
                    brightness[(y, x)] = 0
                    continue
                brightness[(y, x)] = max(0, brightness[y, x] - 1)

    def toggle(intvrl: LightInterval) -> None:
        for y in range(intvrl.y0, intvrl.y1 + 1):
            for x in range(intvrl.x0, intvrl.x1 + 1):
                if part_1:
                    if brightness[(y, x)]:
                        brightness[(y, x)] = 0
                    else:
                        brightness[(y, x)] = 1
                    continue
                brightness[(y, x)] += 2

    on_lights, brightness = defaultdict(set), defaultdict(int)
    actions = {'on': turn_on, 'off': turn_off, 'toggle': toggle}
    intervals = read_input(day=6, parse=parse)

    for interval in intervals:
        actions[interval.action](interval)

    if part_1:
        return sum(bool(v) for v in brightness.values())
    return sum(v for v in brightness.values())


def day_7(part_1=True) -> int:
    def get_value(d: Dict[str, int], k_: str) -> int | None:
        if k_.isdigit():
            return int(k_)
        return d.get(k_, None)

    initializes_graph, init_, values = read_input(day=7,
                                                  delim=None,
                                                  parse=parse_day_7)[0]

    def solve(vals: Dict[str, int]) -> None:
        print(f'{vals=}')
        to_search = {wire for k in vals
                     for wire in initializes_graph.get(k, [])
                     }
        while to_search:
            next_search = set()
            for key_ in to_search:
                gate: LogicGate = init_[key_]
                args_ = tuple(get_value(vals, arg) for arg in gate.args)
                if any(arg is None for arg in args_):
                    continue
                vals[key_] = gate.op_(*args_)
                next_search.update(initializes_graph.get(key_, []))
            to_search = next_search

    values_copy = {**values}
    solve(values)
    if part_1:
        return values['a']
    values_copy['b'] = values['a']
    solve(values_copy)
    return values_copy['a']


def day_8(part_1=True) -> int:
    data, ret = read_input(day=8), 0
    inc_, dont_cnt = 2, {
        r'\\': (1, '-&-'),
        r'\"': (1, ')&('),
        r'\x': (3, '^&^')
    }
    if not part_1:
        inc_, dont_cnt = 0, {
            r'"': (2, '-&-'),
            r'\\': (2, ')&('),
            r'\x': (1, '^&^')
        }

    for line in data:
        ret += inc_
        for dnc, (w, spcl_chr) in dont_cnt.items():
            line = line.replace(dnc, spcl_chr)
            ret += line.count(spcl_chr) * w
    return ret


def day_9(part_1=True) -> int:
    def parse(data: str) -> defaultdict:
        grph = defaultdict(list)
        for ln in data.strip().split('\n'):
            cities, distance = (s.strip() for s in ln.split('='))
            start, stop = (s.strip() for s in cities.split('to'))
            heapq.heappush(grph[start], Edge(int(distance), stop))
            heapq.heappush(grph[stop], Edge(int(distance), start))
        return grph

    def modified_prims(start_city: str) -> int:
        graph_: defaultdict = copy.deepcopy(graph)
        used, current_city, cost = set(), start_city, 0
        while current_city is not None:
            used.add(current_city)
            edge = Edge(0, current_city)
            while graph_[current_city] and \
                    (edge := heapq.heappop(
                        graph_[current_city])).vertex in used:
                pass
            if edge.vertex in used:
                current_city = None
                continue
            cost += edge.wght
            current_city = edge.vertex
        return cost if used == graph_.keys() else float('inf')

    graph = read_input(day=9, delim=None, parse=parse)[0]
    return min(modified_prims(city) for city in graph)


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
