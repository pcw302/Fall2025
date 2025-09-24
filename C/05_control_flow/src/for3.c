#include <stdio.h>

int main()
{
  int i = 0;
  printf("Just before the loop, i = %d.\n", i);

  for (int i = 1; i <= 5; i++)  {  /* The i declared here is local to the loop. */
    printf("Inside the loop, i = %d.\n", i);
  }

  printf("Just after the loop, i = %d.\n", i);
}
