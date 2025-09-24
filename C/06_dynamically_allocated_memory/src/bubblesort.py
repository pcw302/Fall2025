def bubblesort(a):
    n = len(a)
    not_done = True

    while not_done:
        not_done = False
        for i in range(n-1):   # Sweep from left to right.
            if a[i] > a[i+1]:  # Swap adjacent out-of-order terms.
                a[i], a[i+1] = a[i+1], a[i]
                not_done = True

a = [5, 4, 3, 2, 1]

print(f'before bubblesort {a = }')

bubblesort(a)

print(f'after bubblesort  {a = }')
