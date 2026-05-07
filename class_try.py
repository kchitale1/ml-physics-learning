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
    
if __name__=="__main__":
    v = Vector2D(3,4) 
    w = Vector2D(4,1)
    ad = v.add(w)
    sc = v.scale(5)
    print("mag: ", v.magnitude())
    print("add: ", ad.x, ad.y)
    print("scale: ", sc.x, sc.y)
    print("subtract: ", v.subtract(w).x, v.subtract(w).y)
    print("dot: ", v.dot(w))
    print("normalize: ", v.normalize().x, v.normalize().y, v.normalize().magnitude())

    v = Vector2D(3, 4)
    w = Vector2D(1, 2)

    assert repr(v) == "Vector2D(3, 4)"
    assert (v + w).x == 4 and (v + w).y == 6
    assert len(v) == 2
    assert v[0] == 3 and v[1] == 4

    # bonus: iteration works for free
    assert list(v) == [3, 4]

    print("All Day B assertions passed.")
