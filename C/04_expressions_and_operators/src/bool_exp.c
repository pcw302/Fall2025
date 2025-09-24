#include <stdio.h>

int main(int argc, char **argv)
{
  float x = 42;

  printf("x = %.1f\n", x);
  if (0 <= x <= 1) {
    printf("whoa!  0 <= %.1f <= 1 evaluates to true!\n", x);
  }

  x = -1.0;
  printf("x = %.1f\n", x);
  if ((0 <= x) <= 0.5) {
    printf("ok!  (0 <= %.1f) <= 0.5 evaluates to true!\n", x);
  }

  return 0;
}
