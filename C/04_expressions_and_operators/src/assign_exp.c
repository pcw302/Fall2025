// An illustration of an assignment expression.
#include <stdio.h>
int main(int argc, char **argv)
{
  float x = 42;
 
  printf("x = %.1f\n", x);
  if (x = 54) {  // x = 54 here is like Python's walrus.
    printf("yikes!\n");
  }
  
  return 0;
}
