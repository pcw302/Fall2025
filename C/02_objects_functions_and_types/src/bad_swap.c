#include <stdio.h>

void swap(int a, int b) {
  int t = a;
  a = b;
  b = t;
  printf("inside swap(): a = %d, b = %d\n", a, b);
}
