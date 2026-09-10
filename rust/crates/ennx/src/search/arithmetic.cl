// Finite IEEE binary64 arithmetic, round-to-nearest, ties-to-even.
// Integer operations avoid backend-specific floating-point contraction and precision.
typedef unsigned long Word;
typedef unsigned int Bits;

Word shift_jam(Word value, int bits) {
    if (bits <= 0) return value;
    if (bits >= 64) return value != 0;
    return (value >> bits) | ((value << (64 - bits)) != 0);
}

int word_exp(Word value) {
    int exponent = (int)((value >> 52) & 2047);
    return exponent == 0 ? -1022 : exponent - 1023;
}

Word word_sig(Word value) {
    Word fraction = value & 0xfffffffffffffUL;
    return ((value >> 52) & 2047) == 0 ? fraction : fraction | (1UL << 52);
}

Word word_pack(Word sign, int exponent, Word significand) {
    if (significand == 0) return sign;
    while (significand >= (1UL << 56)) {
        significand = shift_jam(significand, 1);
        exponent++;
    }
    while (significand < (1UL << 55) && exponent > -1022) {
        significand <<= 1;
        exponent--;
    }
    if (exponent < -1022) {
        significand = shift_jam(significand, -1022 - exponent);
        exponent = -1022;
    }
    Word tail = significand & 7;
    Word rounded = (significand >> 3) + (tail > 4 || (tail == 4 && ((significand >> 3) & 1)));
    if (rounded >= (1UL << 53)) {
        rounded >>= 1;
        exponent++;
    }
    if (exponent > 1023) return sign | 0x7ff0000000000000UL;
    Word encoded = rounded < (1UL << 52) ? 0 : (Word)(exponent + 1023);
    return sign | (encoded << 52) | (rounded & 0xfffffffffffffUL);
}

bool word_less(Word left, Word right) {
    if (((left | right) << 1) == 0) return false;
    Word sign = 1UL << 63;
    if ((left ^ right) & sign) return (left & sign) != 0;
    return (left & sign) ? left > right : left < right;
}

Word word_max(Word left, Word right) {
    return word_less(left, right) ? right : left;
}

Word word_min(Word left, Word right) {
    return word_less(left, right) ? left : right;
}

Word word_add(Word left, Word right) {
    Word magnitude = 0x7fffffffffffffffUL;
    if ((left & magnitude) < (right & magnitude)) {
        Word temporary = left; left = right; right = temporary;
    }
    int exponent = word_exp(left);
    Word a = word_sig(left) << 3;
    Word b = shift_jam(word_sig(right) << 3, exponent - word_exp(right));
    Word sign = left & (1UL << 63);
    if ((left ^ right) & (1UL << 63)) {
        if (a == b) return 0;
        return word_pack(sign, exponent, a - b);
    }
    return word_pack(sign, exponent, a + b);
}

Word word_mul(Word left, Word right) {
    Word a = word_sig(left), b = word_sig(right);
    Word sign = (left ^ right) & (1UL << 63);
    if (a == 0 || b == 0) return sign;
    int exponent = word_exp(left) + word_exp(right);
    while (a < (1UL << 52)) { a <<= 1; exponent--; }
    while (b < (1UL << 52)) { b <<= 1; exponent--; }
    Word low = (a & 0xffffffffUL) * (b & 0xffffffffUL);
    Word middle = (a >> 32) * (b & 0xffffffffUL) + (low >> 32);
    Word high = middle >> 32;
    middle = (middle & 0xffffffffUL) + (a & 0xffffffffUL) * (b >> 32);
    high += (a >> 32) * (b >> 32) + (middle >> 32);
    low = (middle << 32) | (low & 0xffffffffUL);
    int bits = (high & (1UL << 41)) ? 50 : 49;
    exponent += bits == 50;
    Word significand = (high << (64 - bits)) | (low >> bits);
    significand |= (low << (64 - bits)) != 0;
    return word_pack(sign, exponent, significand);
}

Bits float_pack(int exponent, Word significand) {
    if (exponent < -126) {
        significand = shift_jam(significand, -126 - exponent);
        exponent = -126;
    }
    Word tail = significand & 7;
    Word rounded = (significand >> 3) + (tail > 4 || (tail == 4 && ((significand >> 3) & 1)));
    if (rounded >= (1UL << 24)) { rounded >>= 1; exponent++; }
    if (exponent > 127) return 0x7f800000U;
    Bits encoded = rounded < (1UL << 23) ? 0 : (Bits)(exponent + 127);
    return (encoded << 23) | ((Bits)rounded & 0x7fffffU);
}

Bits word_float(Word value) {
    Bits sign = (Bits)(value >> 32) & 0x80000000U;
    return sign | float_pack(word_exp(value), shift_jam(word_sig(value), 26));
}

Word float_word(Bits value) {
    Word sign = ((Word)value & 0x80000000UL) << 32;
    Bits exponent = (value >> 23) & 255;
    Word significand = value & 0x7fffffU;
    if (exponent == 255) return sign | 0x7ff0000000000000UL;
    if (exponent != 0) return sign | ((Word)(exponent + 896) << 52) | (significand << 29);
    if (significand == 0) return sign;
    int power = -126;
    while (significand < (1UL << 23)) { significand <<= 1; power--; }
    return sign | ((Word)(power + 1023) << 52) | ((significand & 0x7fffffUL) << 29);
}

// Non-negative numerator and positive finite denominator.
Bits float_div(Bits left, Bits right) {
    if (left == 0 || left == 0x7f800000U) return left;
    Word a = word_sig(float_word(left)), b = word_sig(float_word(right));
    int exponent = word_exp(float_word(left)) - word_exp(float_word(right));
    a >>= 29; b >>= 29;
    if (a < b) { a <<= 1; exponent--; }
    Word numerator = a << 26;
    Word quotient = numerator / b;
    quotient |= (numerator % b) != 0;
    return float_pack(exponent, quotient);
}
