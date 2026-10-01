// R2.2a 激励：bank 路由 + 冲突统计（纯组合）
//
// 对应 rtl/bank_map.sv。没有时钟，所以 TB 只做一件事：
//   摆好这一拍的 NBANK 个请求地址 -> eval() -> 看每个 bank 收到几个
//
// 场景就是 R2 一直在算的那个：一拍发出 32 个读请求
//   req[0..15] = A 的 16 个地址 = i*128   (i = 0..15)
//   req[16..31]= B 的 16 个地址 = j       (j = 0..15)
//
// 手算的预期（关键在 128 % 32 == 0）：
//   A 的 16 个地址全是 32 的整数倍 -> 全落 bank 0
//   B 的 16 个地址是 0..15         -> 散在 bank 0..15
//   所以 bank 0 收到 16 + 1 = 17 个
//
//      bank 0       : 17   <- 最挤，决定总拍数
//      bank 1 ~ 15  : 各 1
//      bank 16 ~ 31 : 0
//
// 这个 17 就是 R2.3 写完之后波形必须对上的预期拍数。

#include "Vbank_map.h"
#include "verilated.h"
#include <cstdio>

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);
  auto *top = new Vbank_map;

  // ---- 摆好这一拍的 32 个请求地址 ----
  for (int i = 0; i < 16; i++) top->req_addr[i] = i * 128; // A[i][0]
  for (int j = 0; j < 16; j++) top->req_addr[16 + j] = j;  // B[0][j]

  top->eval(); // 纯组合逻辑，没有时钟，eval 一次就够

  std::printf("请求 -> bank 号\n");
  for (int i = 0; i < 32; i++)
    std::printf("  req[%2d]  addr=%4d  ->  bank %2d\n", i, (int)top->req_addr[i],
                (int)top->bank_idx[i]);

  std::printf("\n每个 bank 收到几个请求\n  bank:");
  for (int b = 0; b < 32; b++) std::printf("%2d ", b);
  std::printf("\n  cnt :");
  for (int b = 0; b < 32; b++) std::printf("%2d ", (int)top->bank_cnt[b]);
  std::printf("\n");

  // ---- 逐项检查 ----
  bool allPass = true;
  for (int b = 0; b < 32; b++) {
    int exp = (b == 0) ? 17 : (b <= 15 ? 1 : 0);
    if ((int)top->bank_cnt[b] != exp) {
      allPass = false;
      std::printf("  !! bank %d 收到 %d 个，期望 %d 个\n", b,
                  (int)top->bank_cnt[b], exp);
    }
  }

  std::printf("\n%s\n", allPass ? "Pass" : "Failed");
  delete top;
  return allPass ? 0 : 1;
}
