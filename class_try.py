import math

from torch import rand
import matplotlib.pyplot as plt

class Vector2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Vector2D({self.x}, {self.y})"
    
    def __add__(self, other):
        return Vector2D(self.x + other.x, self.y + other.y)
    
    def __len__(self):
        return 2
    
    def __getitem__(self, i):
        if i == 0:
            return self.x
        elif i == 1:
            return self.y
        else:
            raise IndexError(f"Vector2D index out of range: {i}")

    def magnitude(self):
        return (self.x**2+self.y**2)**0.5
    
    def add(self, other):
        return Vector2D(self.x+other.x,self.y+other.y)
    
    def scale(self, k):
        return Vector2D(self.x*k, self.y*k)
    
    def subtract(self, other):
        return Vector2D(self.x-other.x, self.y-other.y)
    
    def dot(self, other):
        return (self.x*other.x+self.y*other.y)
    
    def normalize(self):
        return Vector2D(self.x/self.magnitude(), self.y/self.magnitude())
    
def random_walk_2d(start: Vector2D, step_size: float = 1.0):
    """
    Yield Vector2D positions of a random walk starting at `start`.
    Each step picks a uniformly random angle in [0, 2π) and moves
    `step_size` in that direction. Runs forever — caller decides
    when to stop.
    """
    current = start
    # i=0
    while True:
        yield current
        # i+=1
        # print(f"step {i}")
        theta = rand(1).item() * 2 * math.pi
        step = Vector2D(math.cos(theta) * step_size, math.sin(theta) * step_size)
        current = current.add(step)

def save_walk(positions, path):
    with open(path, "w") as f:
        for p in positions:
            f.write(f"{p.x},{p.y}\n")

def load_walk(path):
    with open(path) as f:        # default mode is "r"
        for line in f:
            x, y = (float(v) for v in line.strip().split(","))
            yield Vector2D(x, y)

if __name__=="__main__":
    v = Vector2D(3,4) 
    w = Vector2D(4,1)

    #Day E exercises: generators
    walker = random_walk_2d(Vector2D(0, 0), step_size=0.1)    
    for _ in range(5):
        print(next(walker))

    from itertools import islice
    positions = list(islice(walker, 100))
    print(positions[:1])

    save_walk(positions, "walk.csv")

    # later — reload lazily, no big list in memory
    loaded = list(load_walk("walk.csv"))
    xs, ys = zip(*((p.x, p.y) for p in loaded))
    plt.plot(xs, ys)
    plt.gca().set_aspect("equal")
    plt.show()
    ## Day B exercises:
    # ad = v.add(w)
    # sc = v.scale(5)
    # print("mag: ", v.magnitude())
    
    # print("add: ", ad.x, ad.y)
    # print("scale: ", sc.x, sc.y)
    # print("subtract: ", v.subtract(w).x, v.subtract(w).y)
    # print("dot: ", v.dot(w))
    # print("normalize: ", v.normalize().x, v.normalize().y, v.normalize().magnitude())

    # v = Vector2D(3, 4)
    # w = Vector2D(1, 2)

    # assert repr(v) == "Vector2D(3, 4)"
    # assert (v + w).x == 4 and (v + w).y == 6
    # assert len(v) == 2
    # assert v[0] == 3 and v[1] == 4

    # # bonus: iteration works for free
    # assert list(v) == [3, 4]

    # print("All Day B assertions passed.")


