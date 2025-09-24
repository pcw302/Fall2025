#include <stdio.h>

#define TRUE 1
#define FALSE 0

void bubblesort(int *a, int n)
{
  int not_done;
  int temp;

  do {
    not_done = FALSE;
    for (int i = 0; i < n-1; i++) {  /* Sweep from left to right. */
      if (a[i] > a[i+1]) {  /* Swap adjacent out-of-order terms. */
        temp = a[i];
        a[i] = a[i+1];
        a[i+1] = temp;
        not_done = TRUE;
      }
    }
  } while (not_done);
}

int main(void)
{
  int n = 5;
  int a[] = {5, 4, 3, 2, 1};

  printf("before bubblesort a = ");
  for (int i = 0; i < 5; i++) {
    printf("%d ", a[i]);
  }
  printf("\n");

  bubblesort(a, n);

  printf("after  bubblesort a = ");
  for (int i = 0; i < 5; i++) {
    printf("%d ", a[i]);
  }
  printf("\n");

  return 0;
}
