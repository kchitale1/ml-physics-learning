Day A:
Learning python class definition and defining methods. How self is used internally to the class. __main__ block guard for when num_workers>0 for the future. Return the output of operations as as instance of the class itself is allowed. Capital letters for class names is convention. 

Day B:
Dunder methods also called as special methods in python. __len__ and __getitem__ important for future on Dataset. __repr__ should be used default instead of __str__ by developers. __getitem__ gives free iteration if implemented correctly. 

Day C:
Inheritance. Defining subclass from a class. Can override original methods. super() used inside child's methods to use parent's method instead of replacing. In Dataset class __len__ and __getitem__ are not defined and have to be defined in the subclass. This is standard in pytorch that main class provides framework and main method has to be defined by the subclass. self is subclass instance even inside the parent methods so polymorphism works. Reflection or introspection: "let me decide what I am at runtime". Method resolution order by doing __mro__. Dataset provides the contract, DataLoader provides the machinery

Day D:
lists, iterables: new_list = [expression for member in iterable] 
To add conditionals: 
new_list = [expression for member in iterable if conditional] or
new_list = [true_expr if conditional else false_expr for member in iterable]
Nesting is possible eg. [n for n i num for let in letters] etc
zip creates tuples that match indices of passed lists. Lists with square brackets, dictionaries with curly brackets. Set has unique values only with curly brackets. Generator with curly brackets. Dictionaries have key:value system than indenxing for lists. Use zip_longest when there is index mismatch. starred unpacking takes everything that is remaining from iterable. Use str.format() for reusable template. f-strings can do arithmatic but need variables present at runtime when the line is executed. Use %s style for logging

Day E:
Generators use yield to pause to save memory. It remembers the state of the function. Can be used in comprehension in curly brackets, similar form in list will use entire memory by building the full list. Use next() to step. A function with yield anywhere in its body becomes a generator. Usually called with a look but yield will make sure the function only progresses when user asks. "with" is for context manager, opening files etc to handle exceptions (analog to try: and finally:) 
with <expression> as <name>:
    <body>  
Calling a generator function does not really execute it but just constructs the object, it only rusn with next(gen_fun) is called upto yield. sum(), list() etc run to exhaustion regardless of number of yields. Thus be careful when having while True in a loop and calling sum() on it. 
islice is iterator slice: islice(iterable, start, stop, step). It returns an iterator and not a list (lazy). Need to be wrapped in list if entire thing is needed. Used in streaming data to get a defined block eg list(islice(generator, 100))

Day F:
Dataset and fake loader
A training-dataset's __getitem__(i) returns everything the loss needs to evaluate one example — input and the correct answer.
for x, y in loader:
A FakeLoader wraps a dataset, and when you iterate it, you get one batch at a time. Each batch is a chunk of batch_size examples, optionally in shuffled order.
DataLoader init gets batch size and shuffle true or fales. iter function iterates over the batch size and builds the lists.
Refer Dataset_learning.py

Day G:
Annotated MINST code. Importance of calling super(), dunders used. How Dataset and DataLoader interact etc. What context manager with no_grad does etc. 

Day 4:
Back to physics AI learning. Building 1D diffusion problem. Defining Dataset class and using pytorch DataLoader. If num_workers>0 for windows, the code needs to be in if __name__=='__main__' guard as forking is not available. 
Dataset has one job: given an index, return one sample. It knows nothing about batches, shuffling, or parallelism. Your DataLoader has one job: take a Dataset and turn it into a stream of batches. It knows nothing about what the data actually is.
zip(*batch) transposes a list of (input, target) pairs into two tuples — one of all inputs, one of all targets. The * is the key: it converts the list into positional arguments so zip can do its column-grouping. ie you get one tensor for all inputs and one for all outputs after stacking. 
Custom collate function can also pad for different sizes. 

