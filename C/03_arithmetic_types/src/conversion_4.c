// In case you're not anxious enough, one more illustration of scary conversions between integer types.
#include <stdio.h>
int main(int argc, char **argv)
{
  signed char schar;
  unsigned int uint;

  printf("Rule 4:\n");  
  schar = -127;
  uint = schar;
  printf("schar = -127; uint = schar results in uint = %u\n", uint);
  
  return 0;
}
