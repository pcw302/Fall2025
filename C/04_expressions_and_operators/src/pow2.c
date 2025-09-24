// Compute powers of 2 using shifts.

n = 1;
printf("%u\n", n);
    
for (int i = 0; i < 32; i++) {
    n = n << 1;
    printf("%u\n", n);
}
