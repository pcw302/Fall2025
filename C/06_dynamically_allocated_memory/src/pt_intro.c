#include <stdio.h>

int main(void)
{
  int n = 42;
  int *p;
  int *q;

  printf("n = %d\n", n);

  p = &n;  /* Set p to point at n. */
  printf("address of n: %p\n", (void*) p);

  printf("p dereferences to: %d\n", *p);  /* Display the integer value p points to. */

  *p = 54;  /* Since p points to n, this changes the value of n! */

  printf("n = %d\n", n);

  q = &n;   /* Both p and q point to n. */
  *q = 42;  /* Change n through q. */

  printf("n = %d\n", n);

  return 0;
}
