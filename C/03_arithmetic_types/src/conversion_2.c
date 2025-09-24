// Another illustration of scary conversions between integer types.  
#include <stdio.h>
int main(int argc, char **argv)
{
  unsigned char uchar;

  printf("Rule 2:\n");  
  uchar = 257;  /* uchar is an 8-bit unsigned char, so 257 it outside its range 0 to 255. */
  printf("uchar = 257 results in uchar = %hhu\n\n", uchar);  /* 257 mod 2**8 = 256. */
  
  printf("Another Rule 2:\n");
  printf("Is -1L < 1U?\n");
  if (-1L < 1U) {  /* -1L is converted to unsigned long int for the comparison. */
    printf("Yes, -1L < 1U\n");
  }
  else {
    printf("Whoa! -1L >= 1U\n");
  }
  printf("As an unsigned int, -1L = %lu.\n\n", -1L);

  printf("Another Rule 2:\n");  
  printf("Is -1L < 1Lu?\n");
  if (-1L < 1LU) {
    printf("Yes, -1L < 1UL\n");
  }
  else {
    printf("Whoa! -1L >= 1UL\n");
  }
  printf("As an unsigned long int, -1L = %lu.\n\n", -1L);
  
  return 0;
}
