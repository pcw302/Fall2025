// Illustration of bit twiddling operations.

#include <limits.h>  // For CHAR_BIT, the number of bits in a byte.
#include <stdio.h>

int main(void)
{
  auto m = 42;  // Type inferred as int from the integer literal on the right-hand side.
  auto n = 54;

  printf("%d & %d: %3d\n", m, n, m & n);
  printf("%d | %d: %3d\n", m, n, m | n);
  printf("%d ^ %d: %3d\n", m, n, m ^ n);  
  printf("    ~%d: %3d\n\n", m, ~m);

  printf("The binary expansion of %d is ", m);
  for (int i = sizeof(m) * CHAR_BIT - 1; i >= 0; i--) {
    int bit = (m >> i) & 1;  // Counting right to left, shift the i-th bit to the end
    printf("%d", bit);       // and get its value by ANDing with 1.
  }
  printf(".\n");
  return 0;
}
