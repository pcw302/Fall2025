// An illustration of AND and OR.
#include <stdio.h>
int main(void)
{
  printf("%d\n", (42 < 54) && (42 > 9));
  printf("%d\n", (42 < 9) || (42 < 54));
  printf("%d\n", !(42 < 54));
  
  return 0;
}
