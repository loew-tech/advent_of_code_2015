from collections import defaultdict
from operator import lshift, or_, and_, inv, rshift
import re
from typing import Tuple, Dict

from classes import LogicGate, ShopItem
from constants import REGEX_INTS


def parse_day_7(data_: str) -> Tuple[defaultdict,
                                     dict[str, LogicGate],
                                     dict[str, int]]:
    initializes_graph, init_, values = defaultdict(list), {}, {}

    operators = {"OR": or_, 'AND': and_, "RSHIFT": rshift,
                 "LSHIFT": lshift, "NOT": inv}

    for ln in data_.strip().split('\n'):
        operation, wire = ln.split('->')
        wire = wire.strip()
        operation = operation.split()
        if len(operation) == 1:
            if operation[0].isdigit():
                values[wire] = int(operation[0])
                continue
            initializes_graph[operation[0]].append(wire)
            init_[wire] = LogicGate(op_=operators["OR"],
                                    args=[operation[0],
                                          operation[0]])
            continue
        if len(operation) == 2:
            if operation[-1].isdigit():
                values[wire] = ~int(operation[-1])
                continue
            initializes_graph[operation[-1]].append(wire)
            init_[wire] = LogicGate(op_=operators['NOT'],
                                    args=[operation[-1]])
            continue
        arg1, op, arg2 = (str_.strip() for str_ in operation)
        if arg1.isdigit() and arg2.isdigit():
            values[wire] = operation[op](int(arg1), int(arg2))
            continue
        initializes_graph[arg1].append(wire)
        initializes_graph[arg2].append(wire)
        init_[wire] = LogicGate(op_=operators[op],
                                args=[arg1, arg2])

    return initializes_graph, init_, values


def day_21_get_shop() -> defaultdict:
    with open('inputs/21_shop.txt') as shop_file:
        data = shop_file.read().split('\n\n')
    shop = defaultdict(list)
    for category in data:
        cat = category.split(':')[0]
        for ln in category.split('\n')[1:]:
            stats = map(int, re.findall(REGEX_INTS, ' '.join(ln.split()[-3:])))
            shop[cat].append(ShopItem(*stats))
    return shop
