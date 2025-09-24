#include <stdio.h>

int main(int argc, char **argv)
{
  int m;
  printf("Signed overflow!\n");
  n = +2147483647;  /* 2**31 - 1 is the maximum for 32 bit signed integers **/
  printf("m     = %d\n", m);  
  m++;  // m = m + 1
  printf("m + 1 = %d\n\n", m);


  unsigned int n;
  printf("No overflow for unsigned int!\n");
  n = +2147483647;
  printf("n     = %u\n", n);
  n++;  // n = n + 1
  printf("n + 1 = %u\n\n", n);

  /* Unsigned integer overflow to 0. */
  printf("Unsigned overflow!\n");  
  n = 4294967295;  /* 2**32 - 1 */
  printf("n     = %u\n", n);
  n++;  // n = n + 1
  printf("n + 1 = %u\n", n);  
 
  return 0;
}
