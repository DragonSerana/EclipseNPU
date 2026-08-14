#ifndef ECLIPSE_DTYPE_H
#define ECLIPSE_DTYPE_H

// fp16/bf16 转换。内存里只有字节，dtype 只存在于读/写边界：MAC 输入级 fp16→fp32，
// 输出级 fp32→fp16（RNE 舍入，一次）。位运算实现，不依赖编译器扩展，RTL 可复用。
// v0.1 只走 fp16；bf16 为 v0.2 备好，接入时换函数即可。

#include <cstdint>
#include <cstring>

namespace eclipse {

// fp16: 1 符号 + 5 指数 + 10 尾数
inline float fp16ToFp32(uint16_t h) {
  const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  const uint32_t exp = (h >> 10) & 0x1Fu;
  uint32_t man = h & 0x3FFu;
  uint32_t bits;
  if (exp == 0) {
    if (man == 0) {
      bits = sign;
    } else {
      int e = -1; // 次正规：左移归一化，同时数出指数偏移
      do {
        man <<= 1;
        ++e;
      } while ((man & 0x400u) == 0);
      man &= 0x3FFu;
      bits = sign | (uint32_t)(112 - e) << 23 | man << 13;
    }
  } else if (exp == 0x1Fu) {
    bits = sign | 0x7F800000u | man << 13;
  } else {
    bits = sign | (uint32_t)(exp + 112) << 23 | man << 13;
  }
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

// 舍入到最近偶数
inline uint16_t fp32ToFp16(float f) {
  uint32_t x;
  std::memcpy(&x, &f, sizeof(x));
  const uint16_t sign = (uint16_t)((x >> 16) & 0x8000u);
  const uint32_t exp = (x >> 23) & 0xFFu;
  const uint32_t man = x & 0x7FFFFFu;

  if (exp == 0xFFu) {
    if (man == 0)
      return (uint16_t)(sign | 0x7C00u);
    return (uint16_t)(sign | 0x7C00u | (man >> 13) | 0x200u);
  }
  if (exp == 0)
    return sign; // fp32 次正规 < 2^-24，全部舍到 0

  const int32_t e = (int32_t)exp - 112;
  if (e >= 31)
    return (uint16_t)(sign | 0x7C00u);
  if (e <= 0) { // fp16 次正规；e < -10 时小于最小次正规一半，舍到 0
    if (e < -10)
      return sign;
    uint32_t m = man | 0x800000u;
    const uint32_t shift = 14u - (uint32_t)e;
    const uint32_t half = 1u << (shift - 1u);
    uint32_t h = m >> shift;
    const uint32_t rem = m & ((1u << shift) - 1u);
    if (rem > half || (rem == half && (h & 1u)))
      h++;
    if (h >= 0x400u)
      return (uint16_t)(sign | 0x0400u);
    return (uint16_t)(sign | h);
  }

  uint32_t h = (uint32_t)e << 10;
  const uint32_t rem = man & 0x1FFFu;
  uint32_t m10 = man >> 13;
  if (rem > 0x1000u || (rem == 0x1000u && (m10 & 1u)))
    m10++;
  if (m10 == 0x400u) {
    h += 0x400u;
    m10 = 0;
  }
  return (uint16_t)(sign | h | m10);
}

// bf16: 1 符号 + 8 指数 + 7 尾数，指数域与 fp32 相同
inline float bf16ToFp32(uint16_t b) {
  const uint32_t bits = (uint32_t)b << 16;
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

inline uint16_t fp32ToBf16(float f) {
  uint32_t x;
  std::memcpy(&x, &f, sizeof(x));
  uint32_t b = x >> 16;
  const uint32_t rem = x & 0xFFFFu;
  if (rem > 0x8000u || (rem == 0x8000u && (b & 1u)))
    b++;
  return (uint16_t)b;
}

// 内存按字节访问，读/写 16bit，避免对齐与 aliasing 问题
inline uint16_t loadU16(const void *p) {
  uint16_t v;
  std::memcpy(&v, p, sizeof(v));
  return v;
}

inline void storeU16(void *p, uint16_t v) { std::memcpy(p, &v, sizeof(v)); }

} // namespace eclipse

#endif
