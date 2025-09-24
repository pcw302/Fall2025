/* An illustration of type errors. */
#include <stdio.h>
int main(int argc, char **argv)
{
  n = 42;  /* n is used without declaration. */
  int m;  
  int m;   /* Redefinition of m. */
  printf("Th-th-th-that's all, folks!\n");
  return 0;
}
