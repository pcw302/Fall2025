#include <stdio.h>

int main(void)
{
  char *s = "boo!";
  printf("s: %p  s + 42: %p\n", (void*) s, (void*) (s + 42));
  printf("(s + 42) - s: %p\n", (void*) ((s + 42) - s));

  return 0;
}
