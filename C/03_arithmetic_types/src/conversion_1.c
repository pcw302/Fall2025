// An illustration of scary conversions between integer types.
#include <stdio.h>
int main(int argc, char **argv)
{
  int i;
  long int n;

  printf("Rule 1:\n");
  printf("i is a signed int.\n\n");

  n = -2147483648;
  i = n;  /* i = -2^31, probably in the range of a 32-bit signed int. */
  printf("i = % ld results in i = % d\n", n, i);      

  n -= 1;
  i = n;  /* i = -2^31 - 1, outside the range of a 32-bit signed int. */
  printf("i = % ld results in i = % d\n", n, i);      

  n = 2147483647;  
  i = n;  /* i = 2^31 - 1, inside the range of a 32-bit signed int. */  
  printf("i = % ld results in i = % d\n", n, i);

  n += 1;
  i = n;  /* i = 2^31, outside the range of a 32-bit signed int. */
  printf("i = % ld results in i = % d\n", n, i);
  
  return 0;
}
