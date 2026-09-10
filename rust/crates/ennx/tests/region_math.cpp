#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include "arithmetic.cl"

double decode(Word bits) {
    double value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

Word encode(double value) {
    Word bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

float decode_float(Bits bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

Bits encode_float(float value) {
    Bits bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

Word random_word(Word &state) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
}

int main() {
    static_assert(sizeof(Word) == sizeof(double));
    Word state = 71;
    for (int i = 0; i < 1000000; i++) {
        Word left = random_word(state), right = random_word(state);
        if (!std::isfinite(decode(left)) || !std::isfinite(decode(right))) continue;
        Bits a = static_cast<Bits>(left) & 0x7fffffffU;
        Bits b = static_cast<Bits>(right) & 0x7fffffffU;
        if (word_float(left) != encode_float(static_cast<float>(decode(left)))) return 2;
        if (std::isfinite(decode_float(a)) && std::isfinite(decode_float(b)) && b != 0) {
            if (float_word(a) != encode(static_cast<double>(decode_float(a)))) return 3;
            if (float_div(a, b) != encode_float(decode_float(a) / decode_float(b))) {
                std::fprintf(stderr, "division mismatch %x %x\n", a, b);
                return 4;
            }
        }
        Word sum = word_add(left, right), product = word_mul(left, right);
        if (sum != encode(decode(left) + decode(right)) ||
            product != encode(decode(left) * decode(right))) {
            std::fprintf(stderr, "mismatch %lx %lx: add %lx expected %lx; mul %lx expected %lx\n",
                left, right, sum, encode(decode(left) + decode(right)),
                product, encode(decode(left) * decode(right)));
            return 1;
        }
    }
    std::puts("1000000 binary64 and binary32 arithmetic cases passed");
}
