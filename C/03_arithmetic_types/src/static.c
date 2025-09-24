// Illustration of the static storage class.
#include <stdio.h>

// n has file scope.
static int n = 42;  /* Initialized to 42 at compile time. */

void foo(void)
{
  printf("In foo():\n"); 
  printf("static int n: %d (file scope)\n\n", n);
}

void bar(int a)
{
  // m will persist between calls to bar().
  static int m = 0;  /* Initialized to 0 at compile time. */
  int n = 9;  /* This local n shadows the static n. */
  
  printf("In bar():\n");
  printf("static int m: %d (function scope)\n", m);
  printf("local  int n: %d (function scope)\n\n", n);
  if (m == 0) {
    m = a;
  }
}

int main(void)
{
  printf("static int n: %d (file scope)\n\n", n);
  
  foo();

  bar(42);
  bar(54);
  bar(54);

  return 0;
}
