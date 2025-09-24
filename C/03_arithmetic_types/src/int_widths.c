// The width (in C bytes) of every flavor of integers.
#include <stdio.h>
int main(int argc, char **argv)
{
    /* sizeof() applied to types. */
    printf("Sizes (in bytes) of different flavors of integers:\n");
    printf("A char is           %lu byte long.\n", sizeof(char));
    printf("A signed char is    %lu byte long.\n", sizeof(signed char));
    printf("An unsigned char is %lu byte long.\n\n", sizeof(unsigned char));

    printf("A short is               %lu bytes long.\n", sizeof(short));
    printf("A short int is           %lu bytes long.\n", sizeof(short int));
    printf("A signed short is        %lu bytes long.\n", sizeof(signed short));        
    printf("A signed short int is    %lu bytes long.\n", sizeof(signed short int));
    printf("An unsigned short is     %lu bytes long.\n\n", sizeof(unsigned short));    
    printf("An unsigned short int is %lu bytes long.\n\n", sizeof(unsigned short int));

    printf("An int is          %lu bytes long.\n", sizeof(int));
    printf("An signed int is   %lu bytes long.\n\n", sizeof(signed int));    
    printf("An unsigned int is %lu bytes long.\n\n", sizeof(unsigned int));

    printf("A long is               %lu bytes long.\n", sizeof(long));    
    printf("A long int is           %lu bytes long.\n", sizeof(long int));
    printf("A signed long is        %lu bytes long.\n", sizeof(signed long));        
    printf("A signed long int is    %lu bytes long.\n", sizeof(signed long int));
    printf("An unsigned long is     %lu bytes long.\n\n", sizeof(unsigned long));    
    printf("An unsigned long int is %lu bytes long.\n\n", sizeof(unsigned long int));

    printf("A long long is               %lu bytes long.\n", sizeof(long long));    
    printf("A long long int is           %lu bytes long.\n", sizeof(long long int));
    printf("A signed long long is        %lu bytes long.\n", sizeof(signed long long));
    printf("A signed long long int is    %lu bytes long.\n", sizeof(signed long long int));
    printf("An unsigned long long is     %lu bytes long.\n\n", sizeof(unsigned long long));    
    printf("An unsigned long long int is %lu bytes long.\n\n", sizeof(unsigned long long int));
    return 0;
}
