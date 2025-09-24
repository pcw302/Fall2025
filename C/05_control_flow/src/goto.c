#include <stdio.h>

int main(void)
{
  printf("Starting!\n");

  goto foo;  /* goto statement */

  printf("boo!\n");
  printf("hoo!\n");

 foo:  /* label */
  printf("We jumped over the other print statements!\n");
  
  return 0;
}
