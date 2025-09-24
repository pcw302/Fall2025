// Variable length arrays (VLA).  This is a surprisingly subtle language feature.
int main(void)
{
  int n = 42;
  int a[n];  /* The same as int a[42], an array of 42 int. */

  a[0] = 54;

  return 0;
}
