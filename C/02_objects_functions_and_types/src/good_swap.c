#include <stdio.h>

void swap(int *a, int *b) {  // The * indicates pointers.
  int t = *a;  // Copy the value of the thing a points to into b.
  *a = *b;  // Copy the value of the thing b points to into a.
  *b = t;
  printf("inside swap(): a = %d, b = %d\n", *a, *b);
}
