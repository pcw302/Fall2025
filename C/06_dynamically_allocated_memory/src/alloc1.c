#include <stdio.h>
#include <stdlib.h>  /* You need this to use malloc() and calloc(). */

int main(void)
{
  int n = 42;
  double *x;

  if ((x = (double*) malloc(n * sizeof(double))) == nullptr) {
    fprintf("Allocation of x failed!\n");
  }
  else {
    free(x);
  }

  return 0;
}
