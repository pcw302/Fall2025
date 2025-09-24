#include <stdio.h>

int main(int argc, char **argv)
{
  int c;
  FILE *fp;
  char filename[] = "muskmelon.txt";

  if ((fp = fopen(filename, "r")) != nullptr) {
    while ((c = getc(fp)) != EOF) {
        putchar(c);
    }
    fclose(fp);
  }
  else {
    printf("File \"%s\" not found!\n", filename);
  }

  return 0;
}
