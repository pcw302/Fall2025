#include <stdio.h>

int main()
{
  int i = 0;
  printf("just before the loop, i = %d\n", i);
  
  for (int i = 1; i <= 5; i++) {
    printf("inside the loop, i = %d\n", i);
  }
  
  printf("just after the loop, i = %d\n", i);
}
