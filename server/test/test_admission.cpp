// Unit tests for should_reject_oversized — pure, GPU-free.
//
// Semantics: reject (return true) ONLY when prompt+max_output > max_ctx
// AND compression is NOT enabled. When compression is enabled, let the
// request through so the post-compress effective-size check is the real gate.
//
// Build:
//   /usr/bin/g++-11 -std=gnu++17 -O0 -g \
//     -I/home/peppi/Dev/lucebox-hub/server/src \
//     -o /tmp/test_admission \
//     /home/peppi/Dev/lucebox-hub/server/test/test_admission.cpp && \
//   /tmp/test_admission
#include "server/admission.h"

#include <cstdio>

static int test_failures = 0;
static int test_count    = 0;

#define TEST_ASSERT(expr) do {                                  \
    test_count++;                                               \
    if (!(expr)) {                                              \
        test_failures++;                                        \
        std::fprintf(stderr, "  FAIL: %s:%d: %s\n",            \
                     __FILE__, __LINE__, #expr);                \
    }                                                           \
} while (0)

#define RUN_TEST(fn) do {                                       \
    std::fprintf(stderr, "  %s ...", #fn);                      \
    int before = test_failures;                                 \
    fn();                                                       \
    std::fprintf(stderr, (test_failures == before) ? " ok\n" : "\n"); \
} while (0)

// Case 1: small prompt, no compression -> accept (false).
static void test_small_prompt_no_compression_accepts() {
    // 100 tokens + 100 output = 200 <= 1024 max_ctx -> accept
    TEST_ASSERT(!should_reject_oversized(100, 100, 1024, false));
}

// Case 2: oversized prompt, no compression -> reject (true).
// This preserves the existing hard-reject for uncompressed overflow.
static void test_oversized_no_compression_rejects() {
    // 900 + 200 = 1100 > 1024 max_ctx, no compression -> reject
    TEST_ASSERT(should_reject_oversized(900, 200, 1024, false));
}

// Case 3: oversized prompt WITH compression -> accept (false).
// This is the NEW behavior: let the request through so compression can
// shrink it; the post-compress check is the real gate.
static void test_oversized_with_compression_accepts() {
    // 167000 + 2048 > 65536 max_ctx, but compression enabled -> accept
    TEST_ASSERT(!should_reject_oversized(167000, 2048, 65536, true));
}

// Case 4: exactly at limit -> accept (false).
// prompt + max_output == max_ctx is NOT oversized.
static void test_exactly_at_limit_accepts() {
    // 1024 + 0 == 1024 <= 1024 -> accept
    TEST_ASSERT(!should_reject_oversized(1024, 0, 1024, false));
    // 512 + 512 == 1024 <= 1024 -> accept
    TEST_ASSERT(!should_reject_oversized(512, 512, 1024, false));
}

// Bonus: exactly one over limit, no compression -> reject.
static void test_one_over_limit_no_compression_rejects() {
    // 1025 > 1024 -> reject
    TEST_ASSERT(should_reject_oversized(1025, 0, 1024, false));
}

// Bonus: exactly one over limit, WITH compression -> accept.
static void test_one_over_limit_with_compression_accepts() {
    TEST_ASSERT(!should_reject_oversized(1025, 0, 1024, true));
}

int main() {
    std::fprintf(stderr, "=== test_admission ===\n");
    RUN_TEST(test_small_prompt_no_compression_accepts);
    RUN_TEST(test_oversized_no_compression_rejects);
    RUN_TEST(test_oversized_with_compression_accepts);
    RUN_TEST(test_exactly_at_limit_accepts);
    RUN_TEST(test_one_over_limit_no_compression_rejects);
    RUN_TEST(test_one_over_limit_with_compression_accepts);
    std::fprintf(stderr, "\n%d tests, %d failures\n", test_count, test_failures);
    return (test_failures == 0) ? 0 : 1;
}
