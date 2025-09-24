void foo(int n) {
  double x[n];
}
  
int main(void)
{
  foo(10000000);
  return 0;
}
