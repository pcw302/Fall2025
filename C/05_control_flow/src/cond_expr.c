#include <stdio.h>

int main(void)
{
  int duration;
  
  duration = 1;
  printf("The duration is %d hour%s.\n", duration, (duration > 1) ? "s" : "");

  duration = 2;
  printf("The duration is %d hour%s.\n", duration, (duration > 1) ? "s" : "");
 
  return 0;
}
