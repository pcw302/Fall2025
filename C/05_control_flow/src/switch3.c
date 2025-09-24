#include <stdio.h>

int main(void)
{
  char c;

  c = '2';
  switch (c) {
    case '0': case '1': case '2': case '3': case '4':
    case '5': case '6': case '7': case '8': case '9':
        printf("c is a digit!\n");
        break;
    case ' ': case '\n': case '\t':
        printf("c is white space!\n");
        break;
    default:
        printf("c is something else!\n");
  }
  
  return 0;
}
