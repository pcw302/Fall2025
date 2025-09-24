#include <stdio.h>

int main(void)
{
  int loop = 1;
  int numbers[10];  // An array of 10 int.
  int *p;

  /* The original loop. */
  for (int i = 0; i < 10; i++) {
    numbers[i] = i * i;
    printf("%d ", numbers[i]);
  }
  printf("\n");
  printf("Done with loop %d.\n", loop++);

  p = numbers;
  while (p < numbers + 10) {
    printf("%d ", *p++);
  }
  printf("\n");
  printf("Done with loop %d, p - numbers = %ld.\n", loop++, p - numbers);

  p = numbers;
  for (p = numbers; p < numbers + 10; p++) {
   printf("%d ", *p);
  }
  printf("\n");
  printf("Done with loop %d, p - numbers = %ld.\n", loop++, p - numbers);

  return 0;
}
