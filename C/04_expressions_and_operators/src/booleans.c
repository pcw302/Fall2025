#include <stdio.h>

int main(int argc, char **argv)
{
  if (1) printf("1 is true!\n");
  if (0) printf("The int 0 is false!\n");
  if (3.14) printf("The double 3.14 is true!\n");
  if ('a') printf("The char 'a' is true!\n");
  if (-1) printf("The int -1 is true!\n");
  if (0.0) printf("The double 0.0 is false!\n");
  return 0;
}
