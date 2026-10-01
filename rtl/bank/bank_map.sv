module bank_map #(
  parameter NBANK = 32,
  parameter AW    = 12,
  parameter BW    = 5
) (
  // 非打包数组：Verilator 会把它们生成为 C 数组，TB 里可以直接 [i] 访问。
  // 打包数组（[NBANK-1:0][AW-1:0]）会生成一个宽向量，TB 里要按位拼，很难写。
  input  logic [AW-1:0] req_addr [0:NBANK-1],  // 一拍最多 NBANK 个请求，各带一个地址
  output logic [BW-1:0] bank_idx [0:NBANK-1],  // 每个请求落在哪个 bank
  output logic [5:0]    bank_cnt [0:NBANK-1]   // 每个 bank 收到几个请求（6 位：最多 32）
);
  always_comb begin
    bank_cnt = '{default: '0};                 // 非打包数组的清零写法
    for (int i = 0; i < NBANK; i++) begin
      bank_idx[i] = req_addr[i][BW-1:0];       // 低 BW 位 = 地址 % NBANK
      bank_cnt[bank_idx[i]]++;                 // 第 bank_idx[i] 个 bank 多一个请求
    end
  end
endmodule
