/* An illustration of the const storage class. */
int main(int argc, char **argv)
{
  const int n = 42;
  n = 54;  // Nope!
  return 0;
}
