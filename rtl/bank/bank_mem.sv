module bank_mem #(
  parameter NBANK = 32,
  parameter DEPTH = 128,
  parameter DW    = 16
) (
  input  logic                     clk,
  input  logic                     rst_n,
  // 读侧：一拍最多 NBANK 个，一个槽对一个 bank，地址是 bank 内地址
  input  logic [$clog2(DEPTH)-1:0] rd_addr [0:NBANK-1],
  output logic [DW-1:0]            rd_data [0:NBANK-1],
  // 写侧：和读侧对称，一拍最多 NBANK 个
  input  logic                     wr_en   [0:NBANK-1],
  input  logic [$clog2(DEPTH)-1:0] wr_addr [0:NBANK-1],
  input  logic [DW-1:0]            wr_data [0:NBANK-1]
);

  generate
    for (genvar b = 0; b < NBANK; b++) begin : g_bank
      sram #(
        .DEPTH(DEPTH),
        .DW   (DW)
      ) u_bank (
        .clk  (clk),
        .rst_n(rst_n),
        .we   (wr_en[b]),
        .waddr(wr_addr[b]),
        .wdata(wr_data[b]),
        .raddr(rd_addr[b]),
        .rdata(rd_data[b])
      );
    end
  endgenerate

endmodule
