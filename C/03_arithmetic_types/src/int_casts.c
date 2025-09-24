#include <stdio.h>

int main(int argc, char **argv)
{
  unsigned char uc;  /*  8 bits */
  unsigned int ui;   /* 32 bits */

  ui = 97;
  uc = (unsigned char) ui;  /* This is a cast.  ui is not altered in any way. */

  printf("Widening conversion:\n");
  printf("ui = %u\n", ui);
  printf("uc = %hhd\n\n", uc);

  /* A narrowing conversion. */
  ui = 1024;
  uc = ui;  /* An implicit conversion. */

  printf("Narrowing conversion):\n");
  printf("ui = %u\n", ui);  
  printf("uc = %hhd\n", uc);  /* Oh, noes!  1024 was outside the range of unsigned char! */
  
  return 0;
}
