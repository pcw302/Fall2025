#include <stdio.h>

int main(int argc, char **argv)
{
  int n = 42;
  
  printf("n: %d\n", n);
  n++;  // n = n + 1
  printf("n: %d\n", n);
  n--;  // n = n - 1
  printf("n: %d\n\n", n);

  printf("Prefix ++ and --\n");
  n = 42;
  printf("n at start: %d\n", n);
  printf("++n: %d\n", ++n);  // Set n = n + 1 and print n.
  printf("++n: %d\n", ++n);  // Set n = n + 1 and print n.
  printf("--n: %d\n", --n);  // Set n = n - 1 and print n.
  printf("--n: %d\n\n", --n);  // Set n = n - 1 and print n.

  printf("Suffix ++ and --\n");  
  n = 42;
  printf("n at start: %d\n", n);
  printf("n++: %d\n", n++);  // Print n and set n = n + 1.
  printf("n++: %d\n", n++);  // Print n and set n = n + 1.
  printf("n--: %d\n", n--);  // Print n and set n = n - 1.
  printf("n--: %d\n", n--);  // Print n and set n = n - 1.
  printf("n at end: %d\n", n);
 
  return 0;
}
