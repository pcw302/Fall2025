#include <stdio.h>

int main(void)
{
  int n;
  
  n = 2;
  switch (n) {
    case 1: printf("* ");
    case 2: printf("** ");    /* Jump to here. */
    case 3: printf("*** ");   /* This statement will also be executed. */
    case 4: printf("**** ");  /* As will this one. */
  }
  printf("\nDone with first switch statement!\n\n");

  n = 7;
  switch (n) {
    case 1: printf("* ");
    case 2: printf("** ");
    case 3: printf("*** ");
    case 4: printf("**** ");
  }
  printf("\nDone with second switch statement!\n\n");  

  n = 7;
  switch (n) {
    case 1: printf("* ");
    case 2: printf("** ");
    case 3: printf("*** ");
    case 4: printf("**** ");
    default: printf("No stars for you!");
  }
  printf("\nDone with third switch statement!\n\n");  

  n = 2;
  switch (n) {
    case 2: printf("** ");
    case 1: printf("* ");
    case 3: printf("*** ");
    case 4: printf("**** ");
  }
  printf("\nDone with fourth switch statement!\n\n");    
  
  return 0;
}
