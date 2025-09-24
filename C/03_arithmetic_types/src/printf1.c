// We can trick printf() into reading the bits of a float as an int.
#include <stdio.h>
int main(int argc, char **argv)
{
  float x = 54.0;
  printf("Interpreted as an int:   %d\n", x);
  printf("Interpreted as an float: %f\n", x);  
  return 0;
}
