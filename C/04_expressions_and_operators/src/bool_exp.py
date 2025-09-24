x = 42.0
print(f"{x = }")
if 0 <= x <= 1:
    print(f"whoa!  0 <= {x} <= 1 evaluates to True!")
else:
    print(f"yay!  0 <= {x} <= 1 evaluates to False!")

x = -1
print(f"{x = }")
if (0 <= x) <= 1:
    print(f"whoa!  (0 <= {x}) <= 1 evaluates to True!")
else:
    print(f"yay!  0 <= {x} <= 1 evaluates to False!")
