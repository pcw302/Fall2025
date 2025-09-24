#include <stdio.h>
#include <stdlib.h>

#include "final_swap.h"

int main(void) {  // The void means no arguments.
  int a = 42;
  int b = 54;
  printf("before swap(): a = %d, b = %d\n", a, b);  
  swap(&a, &b);
  printf("after  swap(): a = %d, b = %d\n", a, b);
  return 0;
}

