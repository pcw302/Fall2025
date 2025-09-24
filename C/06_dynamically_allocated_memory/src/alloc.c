#include <stdio.h>
#include <stdlib.h>  /* You need this to use malloc() and calloc(). */

int main(void)
{
  unsigned int n = 42;
  double *x;
  int *a;

  x = (double*) malloc(n * sizeof(double));  /* Space for n doubles. */
  free(x);

  a = (int*) calloc(n, sizeof(int));  /* Space for n integers. */
  free(a);

  a = calloc(n, sizeof(int));  /* Implicit pointer type conversion! */
  free(a);

  return 0;
}
