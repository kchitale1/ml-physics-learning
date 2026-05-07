class Shape:
    """Base class for all 2D shapes."""

    def __init__(self, name):
        self.name = name

    def area(self):
        raise NotImplementedError("Subclasses must implement area()")

    def describe(self):
        return f"I am a {self.name} with area {self.area():.4f}"

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name!r})"
    
import math

class Circle(Shape):
    def __init__(self, radius):
        super().__init__(name="circle")
        self.radius = radius

    def area(self):
        return math.pi * self.radius ** 2


class Rectangle(Shape):
    def __init__(self, width, height):
        super().__init__(name="rectangle")
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height
    

#Run the actual code
if __name__ == "__main__":
    c = Circle(radius=3)
    r = Rectangle(width=4, height=5)

    # Inherited from Shape (not redefined in subclasses):
    print(c.describe())   # I am a circle with area 28.2743
    print(r.describe())   # I am a rectangle with area 20.0000

    # __repr__ is inherited but uses self.__class__.__name__,
    # so it shows the actual subclass name:
    print(c)              # Circle(name='circle')
    print(r)              # Rectangle(name='rectangle')

    # Each subclass's own area() overrides the base:
    assert abs(c.area() - 28.2743) < 1e-3
    assert r.area() == 20

    # Polymorphism: same code works on different shapes
    for shape in [c, r]:
        print(f"{shape} has area {shape.area():.2f}")

    # Try to instantiate the base directly and call area — it should crash
    try:
        Shape("blob").area()
    except NotImplementedError as e:
        print(f"caught expected error: {e}")

    print("All Day C assertions passed.")