/* Illustration of the char/int duality in C. */
#include <stdio.h>
int main(void)
{
    printf("'0' as an int: %d\n", '0');  /* print '0' as an int */
    printf("'2' as an int: %d\n", '2');  /* print '2' as an int */
    printf("'2' - '0': %d\n", '2' - '0');
    printf("48 as a char: %c\n", 48);  /* print 48 as a character */
    printf("54 as a char: %c\n", 54);  /* print 54 as a character */

    return 0;
}
