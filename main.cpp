#include <iostream>
#include "recognize.hpp"

int main() {

    Recognize rec;
    rec.loadModel();
    rec.loadParam(1);
    rec.test_C_(25);

    return 0;
}