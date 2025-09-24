#include <math.h>  /* For pow(). */
#include <stdio.h>
int main(int argc, char **argv)
{
  double (*fun_pt)(double, double) = pow;
  printf("%e\n", (*fun_pt)(2, 0.5));
}
