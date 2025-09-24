#include <stdio.h>

int main()
{
  for (int i = 1; i <= 5; i++) {  /* i is local to for statement. */
    printf("inside the loop, i = %d\n", i);
  }
  printf("just after the loop, i = %d\n", i);

  {                           
    int j = 42;                  /* j is local to this block. */
  }
  printf("just after the block, j = %d\n", j);

  int k = 54;                    /* k is available at any later point. */
  printf("at the end of the program, k = %d\n", k);  
}
