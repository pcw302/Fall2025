#include <stdio.h>

void swap(int *m, int *n)
{
    int temp;

    temp = *m;  /* Set temp = the value that x points to. */
    *m = *n;    /* Set the int that x points to = the int that y points to. */
    *n = temp;  /* Set the location y points to = the value in temp. */
}

int main(void)
{
    int m = 42, n = 54;

    printf("before the call to swap(): m = %d, n = %d\n", m, n);

    swap(&m, &n);  /* Copies of the addresses are passed. */

    printf("after the call to swap():  m = %d, n = %d\n", m, n);

    return 0;
}
