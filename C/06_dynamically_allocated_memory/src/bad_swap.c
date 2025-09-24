#include <stdio.h>

void swap(int *m, int *n)
{
    int temp;

    temp = *m;
    *m = *n;
    *n = temp;
}

int main(void)
{
    int m = 42, n = 54;

    printf("before the call to swap(): m = %d, n = %d\n", m, n);

    swap(&m, &n);  /* Copies of the addresses are passed. */
    
    printf("after the call to swap():  m = %d, n = %d\n", m, n);
    
    return 0;
}
