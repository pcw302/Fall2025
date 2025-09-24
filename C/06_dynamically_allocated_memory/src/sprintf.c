#include <stdio.h>

int main(void)
{
  char s[13];
  sprintf(s, "%e", 42.0);
  printf("boo!\n");
  printf("%s\n", s);
  return 0;
}
