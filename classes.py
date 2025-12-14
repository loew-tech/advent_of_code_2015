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
