module bank_map #(
  parameter NBANK = 32,
  parameter AW    = 12,
  parameter BW    = 5
) (

  input  logic [AW-1:0] req_addr [0:NBANK-1],  // 一拍最多 NBANK 个请求并行
  output logic [BW-1:0] bank_idx [0:NBANK-1],  
  output logic [5:0]    bank_cnt [0:NBANK-1]   // 每个 bank 收到几个请求，最差就是32个请求全部冲突，所以 这里 用 6bit
);
  always_comb begin
    bank_cnt = '{default: '0};                 
    for (int i = 0; i < NBANK; i++) begin
      bank_idx[i] = req_addr[i][BW-1:0];       
      bank_cnt[bank_idx[i]]++;                 
    end
  end
endmodule
