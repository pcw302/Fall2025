/* Yet another illustration of scary conversions between integer types. */
#include <stdio.h>
int main(int argc, char **argv)
{
  signed char schar;
  unsigned char uchar;
  unsigned int uint;

  printf("Rule 3:\n");
  uchar = 128;  /* uchar can represent 128. */
  schar = 128;  /* signed char is only guaranteed to go from -127 to +127. */
  printf("schar = 128 results in schar = %hhd\n", schar);
  
  return 0;
}
