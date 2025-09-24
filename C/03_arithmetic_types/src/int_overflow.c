// Integer overflow, which the standard calls "undefined behavior".
#include <stdio.h>
int main(int argc, char **argv)
{
  int n;

  /* A quiet integer overflow. */
  n = +2147483647;  /* 2**31 - 1 is the maximum for 32 bit signed integers **/
  printf("n     = %+d\n", n);

  n++;
  printf("n + 1 = %+d\n\n", n);

  /* A noisier integer overflow. */
  n = 4294967295;  /* 2**32 - 1 */
  printf("As a signed int, 4294967295 = %d !\n", n);
 
  return 0;
}
