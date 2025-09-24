#include <stdio.h>

int main(void)
{
  int i;

  i = 0;
  while (i < 10) {
    if (i == 3) {
      printf("Skipping 3!\n");
      i++;
      break;  /* Jump out of the while loop. */
    }
    printf("i: %d\n", i);
    i++;
  }
  printf("\n");

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 4; j++) {
      if (j == 2) {
        printf("Skipping j = 2.\n");
        break;  /* Terminate the inner for loop. */
      }
      printf("i: %d  j: %d\n", i, j);
    }
  }
  
  return 0;
}
