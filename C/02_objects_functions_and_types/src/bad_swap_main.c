#include <stdio.h>
#include <stdlib.h>

void swap(int, int);  // Defined in a separate file.

int main(void) {  // The void means no arguments.
  int a = 42;
  int b = 54;
  printf("before swap(): a = %d, b = %d\n", a, b);  
  swap(a, b);
  printf("after  swap(): a = %d, b = %d\n", a, b);
  return 0;
}

