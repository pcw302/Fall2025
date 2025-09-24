#include <stdio.h>

int main(void)
{
  int i;

  /* A for loop. */
  for (i = 42; i < 54; i++) {
    printf("%d ", i);
  }
  printf("\n");

  /* An equivalent while loop. */
  i = 42;
  while (i < 54) {
	printf("%d ", i);
	i++;
  }
  printf("\n");

  return 0;
}
