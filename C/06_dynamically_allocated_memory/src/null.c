#include <stdio.h>

int main(int argc, char **argv)
{
  int *n = nullptr;

  *n = 42;
  printf("n: %d\n", *n);

  return 0;
}
