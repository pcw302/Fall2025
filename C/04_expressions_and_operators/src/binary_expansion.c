// Print a binary expansion by collecting bits as they fall off the end.
{
    unsigned int n = 4294967295;
    unsigned char bits[32];
    n = 42;
    printf("%u\n", n);

    for (int i = 0; i < 32; i++) {
        bits[31 - i] = n & 1;
//        printf("%u\n", n & 1);
        n = n >> 1;
    }
    for (int i = 0; i < 32; i++) {
        if (i && !(i % 8))
            printf(" ");
        printf("%u", bits[i]);
    }        
    printf("\n");
}
