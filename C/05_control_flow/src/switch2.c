#include <stdio.h>

int main(void)
{
  int n;
  
  n = 2;
  switch (n) {
    case 1: printf("* ");
            break;
    case 2: printf("** ");
            break;
    case 3: printf("*** ");
            break;
    case 4: printf("**** ");
            break;
  }
  
  return 0;
}
