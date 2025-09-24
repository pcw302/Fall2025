int main()
{
  int i = 0;
 start:
  if (i >= 42) goto end;
  i++;
  goto start;
 end:
  i = 0;
}
