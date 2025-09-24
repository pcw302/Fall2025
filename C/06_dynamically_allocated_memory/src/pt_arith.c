#include <stdio.h>

int main(void)
{
  int n = 42;
  int *p;

  p = &n;

  printf("%p\n", p);
  printf("%p\n", p + 1);

  printf("Number of ints  between %p and %p: %ld\n", p, (p + 1), (p + 1) - p);
  printf("Number of bytes between %p and %p: %ld\n", p, (p + 1), ((p + 1) - p) * sizeof(int));

  return 0;
}
