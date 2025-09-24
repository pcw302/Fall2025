#include <stdio.h>

int main(int argc, char **argv)
{
  int n = 42;

  while (n < 42) {  /* Test at top of loop; loop does not execute. */
    printf("Hello from the while loop!\n");
  }

  do {              /* Test at bottom of loop; loop executes once. */
    printf("Hello from the do loop!\n");
  } while (n < 42);
  
  return 0;
}
