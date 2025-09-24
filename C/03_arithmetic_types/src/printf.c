// Python borrows C's printf() format codes.
#include <stdio.h>
int main(int argc, char **argv)
{
  float x = 54.0;
  printf("%f\n", x);  // Python: print(f"{x:f}")
  return 0;
}
