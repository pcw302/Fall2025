n = 42 # File scope.

def foo():
    print(n)

def bar():
    n = 54  # Function scope.
    print(n)

foo()
bar()

