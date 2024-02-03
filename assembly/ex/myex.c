#include <stdio.h>
#include <string.h>

int get_num(const char* y) {
    int x;
    x  = strlen(y);
    x += 2;
    return x;
}


int main(int argc, char** argv) {
    const char* y = "hello";
    int x = get_num(y);
    printf("x = %d\n", x);
}
