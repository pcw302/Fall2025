#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
  int *n = (int*) malloc(42 * sizeof(int));

  free(n + 3);  /* n no longer points to the start of the buffer. */

  return 0;
}
