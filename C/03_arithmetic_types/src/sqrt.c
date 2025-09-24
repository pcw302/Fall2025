// An illustration of using the C constexpr evaluation procedure to compute sqrt(2)
// to at least six digits at **compile time**.  A peek at the assembly code will
// confirm sqrt(2) is being calculated.

#include <math.h>
#include <stdio.h>

double rml_abs(double a)
{
  return (a < 0.0) ? -a : a;
}

// Trick the compiler into running several iterations of Newton's method.
const double rml_sqrt(double a, double x)
{
  return rml_abs(x - a/x) <= 1.0e-06 ? x : rml_sqrt(a, 0.5*(x + a/x));
}

int main (void)
{
  constexpr auto r = rml_sqrt(2.0, 1/2.0);
  const double sqrt2 = 1.41421356237309504880;  // For comparison.

  printf("my sqrt(2): %f\n", r);
  printf("   sqrt(2): %f\n", sqrt2);
  
  return 0;
}
