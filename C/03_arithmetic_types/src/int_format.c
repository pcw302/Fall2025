// The same bit pattern interpreted as a signed int and an unsigned int.
#include <stdio.h>
int main(int argc, char **argv)
{
  int n = -1;

  /* %d is the format code for signed int */
  printf("The bits in n interpreted as a signed int: %d\n", n);

  /* %u is the format code for unsigned int */  
  printf("The bits in n interpreted as an unsigned int: %u\n", n);
  
  return 0;
}
