#include <stdint.h>
#include <stdio.h>

int main(void)
{
  printf("BITINT_MAXWIDTH = %d\n\n", BITINT_MAXWIDTH);

  _BitInt(96) n = 42;
}
