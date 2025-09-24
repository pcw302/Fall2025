// Illustration of pointer arithmetic.
#include <stdio.h>

int main(void)
{
  int a[42];
  int b[54];
  double x[6];
  int *p, *q;

  a[3] = 42;

  p = a;
  q = a + 3;  // 3 ints past the start of a[ ].
  printf("q - p = %ld\n", q - p);
  printf("a[3] = %d = *(a+3) = %d = *q = %d\n", a[3], *(a+3), *q);

  // The following is technically not allowed by the standard since
  // p and q point to locations in different arrays.  Nevertheless 
  // it works.
  printf("b - a = %ld\n\n", b - a);

  b[54] = 42;  // Oops!  I'm writing past the end of the array b[ ]!!
  printf("a[0] = %d\n", a[0]);
  
  return 0;
}
