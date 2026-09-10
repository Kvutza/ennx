typedef struct {
    Word length;
    Word initial;
    Word minimum;
    Word maximum;
    Word significant;
    Word lowest;
    Word highest;
    Word observations;
    Word successes;
    Word failures;
    Word tolerance;
    Word initialized;
    Word best;
    Word accepted;
    Word restarted;
    Word restarts;
} Region;

// Layout matches the Rust state; the final four words are host-visible decisions.
void adapt_region(DEVICE const Region *current, DEVICE Region *next, Word value, Word pending) {
    *next = *current;
    next->accepted = word_less(current->best, value); // accepted incumbent
    next->restarted = 0; // restart this update
    next->best = word_max(current->best, value);
    if (current->initialized == 0) {
        next->significant = next->best;
        next->initialized = 1;
    } else {
        Word range = word_add(current->highest, current->lowest ^ (1UL << 63));
        Word scale = word_max(range, 0x3eb0c6f7a0b5ed8dUL); // binary64 1e-6
        Word threshold = word_add(current->significant, word_mul(0x3f50624dd2f1a9fcUL, scale));
        bool improved = word_less(threshold, value);
        next->successes = improved ? current->successes + 1 : 0;
        next->failures = improved ? 0 : current->failures + 1;
        if (improved) next->significant = word_max(current->significant, value);
        if (next->successes >= 3) {
            next->length = word_min(word_mul(current->length, 0x4000000000000000UL), current->maximum);
            next->successes = 0;
        } else if (next->failures >= current->tolerance) {
            next->length = word_mul(current->length, 0x3fe0000000000000UL);
            next->failures = 0;
        }
    }
    next->lowest = word_min(current->lowest, value);
    next->highest = word_max(current->highest, value);
    next->observations = current->observations + 1;
    if (pending == 0 && word_less(next->length, current->minimum)) {
        next->length = current->initial;
        next->successes = 0;
        next->failures = 0;
        next->initialized = 0;
        next->restarted = 1;
        next->restarts = current->restarts + 1;
    }
}
