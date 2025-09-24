int main(void)
{
  int i = 0;
  if (i <= 42) goto foo;
  i = 42;
  goto bar;
 foo:
  i = 54;
 bar:
  i = 6;
}
