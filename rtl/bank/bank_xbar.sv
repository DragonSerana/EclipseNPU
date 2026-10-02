module bank_xbar #(
  parameter NBANK = 32,
  parameter DEPTH = 128,
  parameter DW    = 16,
  // 地址可以拆成低位BW表示bank，和高位LAW表示深度
  parameter BW    = $clog2(NBANK),
  parameter LAW   = $clog2(DEPTH),
  parameter AW    = BW + LAW
) (
  input  logic          clk,
  input  logic          rst_n,
  // 仅 R2.4 对照实验用：1 = bank 号改用 XOR 散列。真实设计里这是设计期决定的，不会是端口
  input  logic          hash_en,
  input  logic [AW-1:0] req_addr [0:NBANK-1],
  output logic [DW-1:0] rd_data  [0:NBANK-1],
  output logic          grant    [0:NBANK-1],
  output logic [5:0]    bank_cnt [0:NBANK-1],
  // 写侧：和读侧对称，一拍最多 NBANK 个请求，各带 valid
  input  logic [AW-1:0] wr_addr    [0:NBANK-1],
  input  logic [DW-1:0] wr_data    [0:NBANK-1],
  input  logic          wr_valid   [0:NBANK-1],
  output logic          wr_grant   [0:NBANK-1],
  output logic [5:0]    wr_bank_cnt[0:NBANK-1]
);

  logic [BW-1:0]  bank_of[0:NBANK-1];
  logic [LAW-1:0] local_of[0:NBANK-1];
  logic           found[0:NBANK-1];
  logic           served[0:NBANK-1];
  logic [BW-1:0]  bank_of_d[0:NBANK-1];

  logic [BW-1:0]  wbank_of[0:NBANK-1];
  logic [LAW-1:0] wlocal_of[0:NBANK-1];
  logic           wfound[0:NBANK-1];

  logic [LAW-1:0] b_rd_addr[0:NBANK-1];
  logic [DW-1:0]  b_rd_data[0:NBANK-1];
  logic           b_we[0:NBANK-1];
  logic [LAW-1:0] b_waddr[0:NBANK-1];
  logic [DW-1:0]  b_wdata[0:NBANK-1];

  // 根据req_addr算出bank和深度，同时记录这个bank要访问的次数
  always_comb begin
    bank_cnt = '{default: '0};
    for (int i = 0; i < NBANK; i++) begin
      bank_of[i] = hash_en ? (req_addr[i][BW-1:0] ^ req_addr[i][2*BW-1:BW])
                           :  req_addr[i][BW-1:0];
      local_of[i] = req_addr[i][AW-1:BW];
      bank_cnt[bank_of[i]]++;
    end
  end

  always_comb begin
    for (int i = 0; i < NBANK; i++) served[i] = 1'b0;
    for (int b = 0; b < NBANK; b++) begin
      found[b]     = 1'b0;
      b_rd_addr[b] = '0;
    end

    // 这里两个循环都用的NBANK，但是表达意思不同，内层i是对NBANK个指令的循环，外层b才是对那个bank的循环
    for (int b = 0; b < NBANK; b++)
      for (int i = 0; i < NBANK; i++)
        // 从某一个bank出发，轮训所有指令，获取他所在bank，如果命中，标记found(标记这个bank )/served(标记这个指令可以用)。
        // 同时b_rd_addr以bank的视角存放这个指令要访问的这个bank的深度，这里其实就标记了b_rd_addr不能bank冲突
        if (!found[b] && bank_of[i] == b[BW-1:0]) begin
          found[b]     = 1'b1;
          served[i]    = 1'b1;
          b_rd_addr[b] = local_of[i];
        end
  end

  always_ff @(posedge clk) begin
    if (!rst_n)
      for (int i = 0; i < NBANK; i++) begin
        grant[i]     <= 1'b0;
        bank_of_d[i] <= '0;
      end
    else
      // 上升沿到来，记录这32条指令，没有bank冲突，放在grant(标记这个指令可以用)和bank_of_d(用的那个bank)
      for (int i = 0; i < NBANK; i++) begin
        grant[i]     <= served[i];
        bank_of_d[i] <= bank_of[i];
      end
  end

  always_comb
    // 读取第(bank_of_d[i])bank数据并输出rd_data
    for (int i = 0; i < NBANK; i++) rd_data[i] = b_rd_data[bank_of_d[i]];

  // 写侧：同样的"按 bank 挑一个"，但不打拍。
  // 读要打拍是因为数据要等 sram 一拍；写在这一拍的沿上就完成了，没有数据要等。
  always_comb begin
    wr_bank_cnt = '{default: '0};
    for (int i = 0; i < NBANK; i++) begin
      wr_grant[i]  = 1'b0;
      wbank_of[i]  = hash_en ? (wr_addr[i][BW-1:0] ^ wr_addr[i][2*BW-1:BW])
                             :  wr_addr[i][BW-1:0];
      wlocal_of[i] = wr_addr[i][AW-1:BW];
      if (wr_valid[i]) wr_bank_cnt[wbank_of[i]]++;
    end

    for (int b = 0; b < NBANK; b++) begin
      wfound[b]  = 1'b0;
      b_we[b]    = 1'b0;
      b_waddr[b] = '0;
      b_wdata[b] = '0;
    end

    for (int b = 0; b < NBANK; b++)
      for (int i = 0; i < NBANK; i++)
        if (!wfound[b] && wr_valid[i] && wbank_of[i] == b[BW-1:0]) begin
          wfound[b]   = 1'b1;
          wr_grant[i] = 1'b1;
          b_we[b]     = 1'b1;
          b_waddr[b]  = wlocal_of[i];
          b_wdata[b]  = wr_data[i];
        end
  end

  bank_mem #(
    .NBANK(NBANK),
    .DEPTH(DEPTH),
    .DW   (DW)
  ) u_mem (
    .clk(clk),
    .rst_n(rst_n),
    // b_rd_addr存放不冲突的bank和深度，组合起来就是，在不冲突的情况下，可以读取的地址
    .rd_addr(b_rd_addr),
    .rd_data(b_rd_data),
    .wr_en(b_we),
    .wr_addr(b_waddr),
    .wr_data(b_wdata)
  );

endmodule
