from collections import namedtuple


# Box = namedtuple('Box', ['l', 'w', 'h'])

class Box:

    def __init__(self, l, w, h: int):
        self.l, self.w, self.h = sorted([l, w, h])

    @property
    def surface_area(self):
        return 2 * self.l * self.w + 2 * self.w * self.h + 2 * self.h * self.l

    @property
    def volume(self):
        return self.l * self.w * self.h

    @property
    def wrapping_paper(self):
        return self.surface_area + self.l * self.w

    @property
    def ribbon(self):
        return 2 * (self.l + self.w) + self.volume


LightInterval = namedtuple('LightInterval', ['action', 'x0', 'y0', 'x1', 'y1'])

LogicGate = namedtuple('LogicGate', ['op_', 'args'])

Edge = namedtuple('Edge', ['wght', 'vertex'])


class Reindeer:

    def __init__(self, pace, duration, rest):
        self.pace, self.duration, self.rest = pace, duration, rest
        self.distance = 0
        self.resting, self.next_rest = 0, self.duration
        self.period_time, self.is_resting = 0, False

    def move(self) -> int:
        self.period_time += 1
        if self.is_resting:
            if self.period_time == self.rest:
                self.period_time = 0
                self.is_resting = False
            return self.distance

        self.distance += self.pace
        if self.period_time == self.duration:
            self.period_time = 0
            self.is_resting = True
        return self.distance

    def __repr__(self):
        return f'Reindeer(pace={self.pace}, duration={self.duration}, ' \
               f'rest={self.rest})'


Ingredient = namedtuple('Ingredient', ['capacity', 'durability', 'flavor',
                                       'texture', 'calories'])
