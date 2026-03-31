#ifndef __RISCV_COREV_ELW_H
#define __RISCV_COREV_ELW_H
#include <stdint.h>
#if defined(__cplusplus)
extern "C" {
#endif
#if defined(__riscv_xcvelw)
#define __DEFAULT_FN_ATTRS \
  __attribute__((__always_inline__, __nodebug__, __artificial__))

// cv.elw rd, imm(rs1): event load word
static __inline__ int32_t __DEFAULT_FN_ATTRS
__riscv_cv_elw_elw(int32_t *ptr) {
  return __builtin_riscv_cv_elw_elw(ptr);
}
#endif
#if defined(__cplusplus)
}
#endif
#endif
