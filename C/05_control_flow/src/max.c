#include <stdio.h>

int main(int argc, char **argv)
{
  int a = 42;
  int b = 54;
  int m;

  m = (a > b) ? a : b;
  printf("max(%d, %d) = %d\n", a, b, m);
  
  return 0;
}
