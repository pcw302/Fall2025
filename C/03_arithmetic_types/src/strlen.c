// An application of sizeof() to strings.
#include <stdio.h>
#include <strings.h>
int main(void)
{
  char s[] = "a";  // String as an array of characters.

  printf("sizeof(s): %lu bytes\n", sizeof(s));
  printf("strlen(s): %lu bytes\n", strlen(s));
  
  return 0;
}
