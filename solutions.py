import math
import time
from collections import defaultdict
from functools import reduce, cache
from hashlib import md5
from itertools import permutations
from json import loads
import heapq
import inspect
import re
import sys
from typing import List, Dict, Set, Tuple

from classes import Box, LightInterval, LogicGate, Edge, Reindeer, Ingredient, \
    Boss, ShopItem
from constants import CARDINAL_DIRECTIONS, DIRECTIONS, REGEX_WORDS, REGEX_INTS
from dbg_utils import *
from helpers import parse_day_7, day_21_get_shop
from utils import read_input, get_inbounds


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
                                                  parse=parse_day_7)

    def solve(vals: Dict[str, int]) -> None:
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


def day_9(part_1=True) -> int | float:
    def parse(data: str) -> defaultdict:
        grph = defaultdict(list)
        sign_ = 1 if part_1 else -1
        for ln in data.strip().split('\n'):
            cities, distance = (s.strip() for s in ln.split('='))
            start, stop = (s.strip() for s in cities.split('to'))
            heapq.heappush(grph[start], Edge(sign_ * int(distance), stop))
            heapq.heappush(grph[stop], Edge(sign_ * int(distance), start))
        return grph

    min_cost = float('inf')

    def dfs(city_: str, used: Set[str], cost: int) -> None:
        if used == graph.keys():
            nonlocal min_cost
            min_cost = min(min_cost, cost)
            return

        for edge in graph[city_]:
            if edge.vertex in used:
                continue
            used.add(edge.vertex)
            dfs(edge.vertex, used, cost + edge.wght)
            used.remove(edge.vertex)

    graph: defaultdict = read_input(day=9, delim=None, parse=parse)
    sign = 1 if part_1 else -1
    for city in graph:
        dfs(city, {city}, 0)
    return sign * min_cost


def day_10(part_1=True) -> int:
    value = read_input(day=10, delim=None)

    def iterate(val: str) -> str:
        new_val, v, cnt = [], '', ''
        for c in val:
            if not c == v:
                new_val.append(f'{cnt}{v}')
                v, cnt = c, 0
            cnt += 1
        return f'{"".join(new_val)}{cnt}{v}'

    iterations = 40 if part_1 else 50
    while (iterations := iterations - 1) >= 0:
        value = iterate(value)
    return len(value)


def day_11(part_1=True) -> str:
    def increment(password_: List[int]) -> List[int]:
        new_ = [*password_]
        new_[-1] += 1
        carry, index = new_[-1] > z, len(new_) - 1
        while carry:
            new_[index] = a
            index -= 1
            new_[index] += 1
            carry = new_[index] > z
        return new_

    data, a, z = read_input(day=11, delim=None), ord('a'), ord('z')
    data = [ord(c) for c in data]

    def solve(password):
        invalid = True
        bad_chars = {ord('i'), ord('o'), ord('l')}
        while invalid:
            password = increment(password)
            pair, run, bad_chr = False, False, False
            pprev, prev, pairs = -1, -1, 0
            for i in password:
                pair += prev == i and not (pprev, prev) == (prev, i)
                run |= pprev + 2 == prev + 1 == i
                bad_chr |= i in bad_chars

                pprev, prev = prev, i
            invalid = pair < 2 or not run or bad_chr
        return password

    pw = solve(data)
    if part_1:
        return ''.join([chr(i) for i in pw])
    return ''.join([chr(i) for i in solve(pw)])


def day_12(part_1=True) -> int:
    data = read_input(day=12, delim=None)

    def solve(content: int | str | List[any] | Dict[str, any]) -> int:
        if type(content) == int:
            return content
        if type(content) == list:
            return sum([solve(con) for con in content])
        if type(content) == str:
            return 0
        if not part_1 and 'red' in content.values():
            return 0
        return solve(list(content.values()))

    return solve(loads(data))


def day_13(part_1=True) -> int:
    def parse(data: str) -> Dict[str, Dict[str, int]]:
        graph_ = defaultdict(dict)
        for ln in data.strip().split('\n'):
            wght = next(map(int, re.findall(r'\d+', ln)))
            sign = 1 if 'gain' in ln else -1
            src, *_, dest = ln.split()
            graph_[src][dest[:-1]] = sign * wght
        return graph_

    def get_happiness(perm: str) -> int:
        happiness = 0
        for i, person in enumerate(perm):
            happiness += graph[perm[i - 1]][person]
            happiness += graph[perm[(i + 1) % len(graph)]][person]
        return happiness

    graph: defaultdict = read_input(day=13, delim=None, parse=parse)

    if not part_1:
        for k in [*graph]:
            graph['-'][k] = 0
            graph[k]['-'] = 0

    perms = permutations(graph.keys(), len(graph.keys()))
    return max(get_happiness(p) for p in perms)


def day_14(part_1=True) -> int:
    def parse(ln: str) -> Reindeer:
        ints = map(int, re.findall(r'\d+', ln))
        return Reindeer(*ints)

    reindeer: List[Reindeer] = read_input(day=14, parse=parse)

    pts, max_distance, time = defaultdict(int), -1, 0
    while (time := time + 1) <= 2503:
        winning_reindeer = []
        for r in reindeer:
            distance = r.move()
            if not winning_reindeer:
                max_distance = distance
                winning_reindeer.append(r)
                continue
            if distance >= (dst := winning_reindeer[0].distance):
                if distance > dst:
                    max_distance = distance
                    winning_reindeer = [r]
                else:
                    winning_reindeer.append(r)
        for rdr in winning_reindeer:
            pts[rdr] += 1

    if part_1:
        return max_distance
    return max(pts.values())


def day_15(part_1=True) -> int:
    ingredients: List[Ingredient] = read_input(day=15,
                                               parse=lambda x: Ingredient(
                                                   *map(int,
                                                        re.findall(r'-?\d+',
                                                                   x))
                                               ))
    keys, max_ = ['capacity', 'durability', 'flavor', 'texture'], -1

    def solve(indx=0, counts_=None):
        if counts_ is None:
            counts_ = defaultdict(int)
            counts_[ingredients[0]] = 100
        nonlocal max_
        max_ = max(max_, get_score(counts_))
        while indx < len(ingredients) - 1 and counts_[ingredients[indx]]:
            counts_[ingredients[indx]] -= 1
            counts_[ingredients[indx + 1]] += 1
            solve(indx + 1, {**counts_})

    def get_score(counts_):
        if not part_1 and \
                not sum(
                    ing.calories * counts_[ing] for ing in ingredients) == 500:
            return -1
        return reduce(lambda x, y: x * y, [
            max(0,
                sum(getattr(ing, field) * counts_[ing] for ing in ingredients))
            for field in keys
        ])

    solve()
    return max_


def day_16(part_1=True) -> int:
    def parse(ln: str) -> Dict[str, int]:
        return dict(zip(re.findall(REGEX_WORDS, ln)[1:],
                        map(int, re.findall(REGEX_INTS, ln)[1:])))

    sues: List[Dict[str, int]] = read_input(day=16, parse=parse)
    target = {
        'children': 3, 'cats': 7, 'samoyeds': 2, 'pomeranians': 3,
        'goldfish': 5, 'trees': 3, 'cars': 2, 'perfumes': 1
    }
    gt, lt = {'cats', 'trees'}, {'pomeranians', 'goldfish'}
    for i, sue in enumerate(sues, start=1):
        if part_1 and all(target.get(key, 0) == v for key, v in sue.items()):
            break

        if not part_1 and \
                all(sue.get(k, float('inf')) >= target[k] for k in gt) and \
                all(sue.get(k, 0) <= target[k] for k in lt) and \
                all(target.get(k, 0) == sue[k] for
                    k in sue.keys() - {*gt, *lt}):
            break
    return i


def day_17(part_1=True) -> int:
    min_ = float('inf')

    @cache
    def solve(amt=0, used=(), target=150) -> None:
        if amt >= target and used not in matches:
            if amt == target:
                nonlocal min_
                min_ = min(min_, len(used))
                matches.add(used)
        for i, b in enumerate(buckets):
            if i in used:
                continue
            solve(amt + b, tuple(sorted((*used, i))), target)

    buckets = sorted(read_input(day=17, parse=lambda x: int(x)), reverse=True)
    matches = set()
    solve()
    return len(matches) if part_1 else sum(len(m) == min_ for m in matches)


def day_18(part_1=True) -> int:
    def is_corner(y_, x_):
        return y_ in {0, len(grid) - 1} and x_ in {0, len(grid[0]) - 1}

    def toggle_lights():
        tmp_grid = [[*row] for row in grid]
        for y, row in enumerate(tmp_grid):
            for x, b in enumerate(row):
                if not part_1 and is_corner(y, x):
                    continue
                on_nghbrs = sum(
                    inbounds(y + yi, x + xi) and tmp_grid[y + yi][x + xi]
                    for yi, xi in DIRECTIONS)
                if b and on_nghbrs not in {2, 3}:
                    grid[y][x] = False
                if not b and on_nghbrs == 3:
                    grid[y][x] = True

    grid = read_input(day=18, parse=lambda x: [xi == '#' for xi in x])
    if not part_1:
        for yy in (0, len(grid) - 1):
            for xx in (0, len(grid[0]) - 1):
                grid[yy][xx] = True
    inbounds = get_inbounds(grid)

    for _ in range(100):
        toggle_lights()
    return sum(sum(v for v in row) for row in grid)


def day_19(part_1=True) -> int:
    def parse(data: str) -> Tuple[defaultdict, str]:
        relations_, chem = data.split('\n\n')
        ret = defaultdict(list)
        for relation in relations_.strip().split('\n'):
            c1, c2 = relation.split('=>')
            ret[c1.strip()].append(c2.strip())
        return ret, chem

    def find_all_occurrences(pattern, str_: str) -> List[int]:
        return [m.start() for m in re.finditer(pattern, str_)]

    def get_all_transformations(molecule) -> Set[str]:
        ret = set()
        indices = find_all_occurrences(molecule, chemical)
        for i in indices:
            ret |= {f'{chemical[:i]}{chemical[i:].replace(molecule, v_, 1)}'
                    for v_ in mappings[molecule]}
        return ret

    mappings, chemical = read_input(day=19, delim=None, parse=parse)
    chemicals = set()
    for mol in mappings:
        chemicals |= get_all_transformations(mol)
    return len(chemicals) if part_1 else NotImplemented


def day_20(part_1=True) -> int:
    def get_presents(num: int) -> int:
        total_presents = 0
        limit = int(num ** .5)
        for i_ in range(1, limit + 1):
            if num % i_ == 0:
                j = num // i_
                if part_1 or j <= 50:
                    total_presents += i_ * (10 + (not part_1))
                if not j == i_ and (part_1 or num // j <= 50):
                    total_presents += j * (10 + (not part_1))
        return total_presents

    target_ = read_input(day=20, delim=None, parse=lambda x: int(x))
    for i in range(1, target_ // 10):
        if get_presents(i) >= target_:
            return i


def day_21(part_1=True) -> int:
    return NotImplemented
    def get_player(equips: List[ShopItem]) -> Boss:
        dmg = sum(item.dmg for item in equips)
        armor = sum(item.armor for item in equips)
        return Boss(hp=100, dmg=dmg, armor=armor)

    def is_winner(equipment_) -> bool:
        player = get_player(equipment_)
        return math.ceil(player.hp // max(1, (boss.dmg - player.armor))) >= \
               math.ceil(boss.hp // max(1, player.dmg - boss.armor))

    boss = Boss(*(map(int, re.findall(REGEX_INTS,
                                      read_input(day=21, delim=None)))))
    shop = day_21_get_shop()
    for k, v in shop.items():
        print(k, v)
    print('\n')

    min_, n = float('inf'), len(shop['Rings'])
    for w in shop['Weapons']:
        equipment = [w]
        print(f'{equipment=}')
        if is_winner(equipment):
            min_ = min(min_, sum(e.cost for e in equipment))
        for a in shop['Armor']:
            no_armor = [equipment[0]]
            equipment.append(a)
            print(f'\t{equipment=}')
            if is_winner(equipment):
                min_ = min(min_, sum(e.cost for e in equipment))
            for i, r in enumerate(shop['Rings']):
                equipment.append(r)
                print(f'\t{equipment=}')
                if is_winner(equipment):
                    min_ = min(min_, sum(e.cost for e in equipment))
                no_armor.append(r)
                print(f'\t{no_armor=}')
                if is_winner(no_armor):
                    min_ = min(min_, sum(e.cost for e in no_armor))
                for j in range(i, n):
                    print('\n----------')
                    equipment.append(shop['Rings'][j])
                    print(f'\t\t{equipment=}')
                    if is_winner(equipment):
                        min_ = min(min_, sum(e.cost for e in equipment))
                    no_armor.append(shop['Rings'][j])
                    print(f'\t\t{no_armor=}')
                    if is_winner(no_armor):
                        min_ = min(min_, sum(e.cost for e in no_armor))
                    equipment.pop()
                    no_armor.pop()
                equipment.pop()
                no_armor.pop()
            equipment.pop()
            no_armor.pop()
        input('break: ')
    return min_


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
        print(f'{day}(part=2) = {funcs[day](part_1=False)}')
