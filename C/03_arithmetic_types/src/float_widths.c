// Widths of floating point numbers.  The __fp16 type is not mandated by the standard.
#include <stdio.h>
int main(int argc, char **argv)
{
  float x;

  /* Width of a variable. */
  printf("The float variable x occupies %lu bytes.\n\n", sizeof(x));
  
  /* Widths (number of C bytes) of different flavors of floating point. */
  printf("A fp16   is %lu bytes.\n", sizeof(__fp16));  // Non-portable gcc extension for ARM/AArch64.
  printf("A float  is %lu bytes.\n", sizeof(float));
  printf("A double is %lu bytes.\n\n", sizeof(double));
  printf("A long double is %lu bytes.\n\n", sizeof(long double));  

  return 0;
}
