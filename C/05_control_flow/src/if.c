#include <stdio.h>

int main(int argc, char **argv)
{
  int n = 42;
  int m = 54;

  /* Use { } even for one-line blocks. */
  if (n > 42) {
    printf("42\n");
  }
  else if (m > 42) {
    printf("54\n");
  }
  else {
    printf("boo!\n");
  }

  /* Legal, but not advisable. */
  if (n > 42)
    printf("42\n");
  else if (m > 42)
    printf("54\n");
  else
    printf("boo!\n");

  // This way of writing one-line blocks is OK with me.
  n =  9 * 6;
  if (n > 42) printf("n: %d\n", n);
 
  return 0;
}
