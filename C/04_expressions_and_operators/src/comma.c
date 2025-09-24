// An illustration of the comma operator.
#include <stdio.h>

int foo(int* n)
{
  return *n += 3;
}

int main(void)
{
  int m = 0;
  int n = 7;
  int i, j, k;

  // The following produces the same result as
  //   m = n++;
  //   n += 2;
  m = n++, n += 2;  // m is now 7.
  printf("m: %d\n", m);
  printf("n: %d\n\n", n);

  m = 0;
  n = 7;

  // The following produces the same result as
  //    n++;
  //    n += 2;
  //    m = n;
  m = (n++, n += 2);  // m is now 10;
  printf("m: %d\n", m);
  printf("n: %d\n\n", n);

  // Here we use the comma operator allow for multiple actions in a for loop.
  for (i = 0, j = 0, k = 0; i < 10; i++, j++, k++);
  
  return 0;
}
