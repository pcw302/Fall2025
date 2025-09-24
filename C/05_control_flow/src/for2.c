#include <stdio.h>

int main(void)
{
  int i;

  for (i = 42; i < 54; i++) {  /* Suffix ++. */
    printf("%d ", i);
  }
  printf("\n");

  for (i = 42; i < 54; ++i) {  /* Prefix ++. */
    printf("%d ", i);
  }
  printf("\n");

  return 0;
}
