#include <stdio.h>

int my_strlen(char *s)
{
    int count;
    for (count = 0; *s; s++, count++);
    return count;
}

int main(void)
{
  char s[] = "Hello, world!";
  char t[] = "You're a lying dog-faced pony soldier!";

  printf("The string '%s' contains %d characters + an ASCII null.\n", s, my_strlen(s));
  printf("The string '%s' contains %d characters + an ASCII null.\n", t, my_strlen(t));
 
  return 0;
}
