#include <stdio.h>

int main(int argc, char **argv)
{
  int c;
  FILE *fp;

  fp = fopen("muskrat.txt", "r");
  while ((c = getc(fp)) != EOF) {
    putchar(c);
  }
  fclose(fp);
     
  return 0;
}
