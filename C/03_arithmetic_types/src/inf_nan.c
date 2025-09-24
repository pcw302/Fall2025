// An illustration of a proper implementation of IEEE-754, unlike Python.
#include <stdio.h>
int main(void)
{
  double inf = 1.0/0.0;
  double nan = 0.0/0.0;

  printf("Here is an Inf: %lf\n", inf);
  printf("Here is a  NaN: %lf\n\n", nan);

  printf("1/inf = %lf\n", 1.0/inf);
  printf("inf - inf = %lf\n", inf - inf);

  if (nan != nan) {
    printf("Yow!  nan != nan!\n");
  }

  return 0;
}
