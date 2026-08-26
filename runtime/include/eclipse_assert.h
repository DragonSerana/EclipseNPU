#ifndef ECLIPSE_ASSERT_H
#define ECLIPSE_ASSERT_H

#include <cstdio>
#include <cstdlib>

namespace eclipse_runtime {

[[noreturn]] inline void eclipse_assert_fail(const char *expr, const char *msg,
                                             const char *file, int line) {
  std::fprintf(stderr, "[EclipseNPU] Assertion failed: %s\n", expr);
  if (msg && msg[0])
    std::fprintf(stderr, "  %s\n", msg);
  std::fprintf(stderr, "  at %s:%d\n", file, line);
  std::abort();
}

} // namespace eclipse_runtime

#ifdef NDEBUG
#define ECLIPSE_ASSERT(cond, msg) ((void)0)
#else
#define ECLIPSE_ASSERT(cond, msg)                                              \
  do {                                                                         \
    if (!(cond))                                                               \
      ::eclipse_runtime::eclipse_assert_fail(#cond, (msg), __FILE__, __LINE__);        \
  } while (0)
#endif

#endif
