#include <stdlib.h>

int main(void)
{
  unsigned int n = 42;
  int *p;

  p = (int*) malloc(n * sizeof(int));
  free(p);
  free(p);

  return 0;
}
