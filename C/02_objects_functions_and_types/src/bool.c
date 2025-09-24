// Booleans in C act like 1 and 0.

#include <stdio.h>

int main(void)
{
  bool T = true;
  bool F = false;

  printf("sizeof(true)  = %lu\n", sizeof(true));
  printf("sizeof(false) = %lu\n", sizeof(false));

  printf("T = %d\n", T);
  printf("F = %d\n", F);

  if (T == 1) printf("T == 1\n");
  if (F == 0) printf("F == 0\n");
 
  return 0;
}
