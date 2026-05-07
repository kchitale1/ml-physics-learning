Day A:
Learning python class definition and defining methods. How self is used internally to the class. __main__ block guard for when num_workers>0 for the future. Return the output of operations as as instance of the class itself is allowed. Capital letters for class names is convention. 

Day B:
Dunder methods also called as special methods in python. __len__ and __getitem__ important for future on Dataset. __repr__ should be used default instead of __str__ by developers. __getitem__ gives free iteration if implemented correctly. 

Day C:
Inheritance. Defining subclass from a class. Can override original methods. super() used inside child's methods to use parent's method instead of replacing. In Dataset class __len__ and __getitem__ are not defined and have to be defined in the subclass. This is standard in pytorch that main class provides framework and main method has to be defined by the subclass. self is subclass instance even inside the parent methods so polymorphism works. Reflection or introspection: "let me decide what I am at runtime". Method resolution order by doing __mro__. Dataset provides the contract, DataLoader provides the machinery