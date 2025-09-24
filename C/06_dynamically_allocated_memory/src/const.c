void foo(const int *a, int *n)
{
  /* const int *a is a pointer to a int where we cannot change
     the value of the int using the pointer. */
   a = n;
  *a = *n;
}

void bar(int* const a, int *n)
{
  /* int* const a is a pointer to a int where we can change the
     value of the int but we cannot change the value of the pointer. */
   a = n;
  *a = *n;
}

void foobar(const int* const a, int *n)
{
  /* const int* const a is a pointer to a int where we can change neither
     the value of the int nor value of the pointer. */  
   a = n;
  *a = *n;
}
