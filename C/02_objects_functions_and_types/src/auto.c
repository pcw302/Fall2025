/* An example of type inference via "auto" in C23. */
int main(int argc, char **argv)
{
  int n = 42;
  auto m = n;  // m is an int.
}
