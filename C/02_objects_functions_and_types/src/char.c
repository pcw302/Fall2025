// An illustration of the character/integer duality in C.

#include <limits.h>  // for CHAR_BIT
#include <stdio.h>

int main(void)
{
  printf("I am the character '%c'.\n\n", 'u');
  char c = 'a';  // Character variable.
  printf("c = '%c'\n\n", c);

  printf("A character is %lu bytes.\n", sizeof(char));
  printf("A byte is %d bytes.\n\n", CHAR_BIT);

  printf("'0' as int: %d\n", '0');   /* print '0' as an int */
  printf("48 as char: '%c'\n\n", 48);  /* print 48 as a character */

  printf("'a' as int: %d\n", 'a');    /* print 'a' as an int */
  printf("'z' as int: %d\n\n", 'z');  /* print 'z' as an int */

  printf("'z' - 'a': %d\n\n", 'z' - 'a');  /* difference as integers */

  printf("%3d as char: %c\n", 97, 97);    // print 97 as an integer and a character
  printf("%3d as char: %c\n", 122, 122);  // print 122 as an integer and a  character
  
  return 0;
}
