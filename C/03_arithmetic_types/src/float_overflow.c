// Floating-point overflow to infinity and underflow to +/-zero.
#include <stdio.h>
int main(void)
{
  double x = 1.0e+308;
  double y = 1.0e-308;
  double z = 1.0/0.0;
  
  printf("%.2e + %.2e + %.2e = %lf\n", x, x, x, x + x + x);
  printf("%.2e / %.2e = %lf\n\n", y, x, y/x);
  printf("1/(+oo) = %+f\n", 1/(+z));
  printf("1/(-oo) = %+f\n", 1/(-z));  
  return 0;
}
