#include <stdio.h>

int main(int argc, char **argv)
{
  printf((42 < 54) ? "true\n" : "false\n");
  printf((42 <= 54) ? "true\n" : "false\n");
  printf((42 == 54) ? "true\n" : "false\n");
  printf((42 != 54) ? "true\n" : "false\n");
  printf((42 >= 54) ? "true\n" : "false\n");
  printf((42 > 54) ? "true\n" : "false\n");    

  return 0;
}
