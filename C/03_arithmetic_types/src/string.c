// Characters and strings.
#include <stdio.h>
int main(void)
{
  char c = 'a';    // Character literal.
  char s[] = "a";  // String as an array of characters.

  printf("sizeof(c): %lu bytes\n", sizeof(c));  // sizeof(char) == 1, always.
  printf("sizeof(s): %lu bytes\n", sizeof(s));
  
  return 0;
}
