#include <stdio.h>

int main(int argc, char **argv)
{
  int c;
  FILE *fp;

  fp = fopen("muskrat.txt", "r");  /* Open the file for reading. */
  c = getc(fp);                    /* Read the first character. */
  while (c != EOF) {               /* Until we hit the END OF FILE { */
    putchar(c);                    /*     print the character to the terminal, */
    c = getc(fp);                  /*     read the next character */
  }                                /* }                           */
 fclose(fp);                       /* Close the file. */
 
  return 0;
}
