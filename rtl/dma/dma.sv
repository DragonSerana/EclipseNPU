module dma #(
  parameter NBANK       = 32,
  parameter DW          = 16,
  parameter DEPTH       = 128,
  parameter BW          = $clog2(NBANK),
  parameter LAW         = $clog2(DEPTH),
  parameter AW          = BW + LAW,
  parameter DDR_LATENCY = 4,
  parameter LAST        = DDR_LATENCY - 1
) (
  input  logic          clk,
  input  logic          rst_n,

  /* verilator lint_off UNUSEDSIGNAL */
  input  logic [31:0]   desc_sram_addr,
  /* verilator lint_on UNUSEDSIGNAL */
  input  logic [31:0]   desc_ddr_addr,
  input  logic [31:0]   desc_rows,
  input  logic [31:0]   desc_cols,
  input  logic [31:0]   desc_src_stride,
  input  logic          desc_start,

  output logic          busy,
  output logic          done,

  output logic [31:0]   ddr_raddr  [0:1],
  output logic          ddr_ren    [0:1],
  input  logic [127:0]  ddr_rdata  [0:1],
  input  logic          ddr_rvalid [0:1],

  output logic [AW-1:0] wr_addr    [0:NBANK-1],
  output logic [DW-1:0] wr_data    [0:NBANK-1],
  output logic          wr_valid   [0:NBANK-1]
);

  typedef enum logic [1:0] { S_IDLE, S_CHECK, S_RUN, S_DONE } state_e;
  state_e state, next;

  // ---------------------------------------------------------------- 描述符
  logic [31:0] row_bytes;
  assign row_bytes = desc_cols << 1;

  logic aligned_ok;
  assign aligned_ok = (desc_sram_addr[3:0] == 4'd0) && (desc_ddr_addr[3:0] == 4'd0);

  // ---------------------------------------------------------------- 搬运状态
  logic [31:0]   st_row;
  logic [31:0]   st_rstart;
  logic [31:0]   st_blk;
  logic [31:0]   st_left;
  logic [AW-1:0] st_wptr;

  // ---------------------------------------------------------------- 突发 A
  logic [3:0]  a_lo;
  logic [4:0]  a_avail, a_take;
  logic [31:0] a_left_raw, a_left_n, a_rstart_n, a_blk_n, a_row_n;
  logic        a_end;

  assign a_lo       = (st_blk == {st_rstart[31:4], 4'd0}) ? st_rstart[3:0] : 4'd0;
  assign a_avail    = 5'd16 - {1'b0, a_lo};
  assign a_take     = (st_left < {27'd0, a_avail}) ? st_left[4:0] : a_avail;
  assign a_left_raw = st_left - {27'd0, a_take};
  assign a_end      = (a_left_raw == 32'd0);

  assign a_left_n   = a_end ? row_bytes : a_left_raw;
  assign a_row_n    = a_end ? (st_row + 32'd1) : st_row;
  assign a_rstart_n = a_end ? (st_rstart + desc_src_stride) : st_rstart;
  assign a_blk_n    = a_end ? ((st_rstart + desc_src_stride) & 32'hFFFFFFF0)
                            : (st_blk + 32'd16);

  // ---------------------------------------------------------------- 突发 B
  logic [3:0]  b_lo;
  logic [4:0]  b_avail, b_take;
  logic [3:0]  b_n_w;
  logic [31:0] b_left_raw, b_left_n, b_rstart_n, b_blk_n, b_row_n;
  logic        b_end;

  assign b_lo    = (a_blk_n == {a_rstart_n[31:4], 4'd0}) ? a_rstart_n[3:0] : 4'd0;
  assign b_avail = 5'd16 - {1'b0, b_lo};
  assign b_take  = (a_left_n < {27'd0, b_avail}) ? a_left_n[4:0] : b_avail;
  assign b_n_w   = b_take[4:1];

  assign b_left_raw = a_left_n - {27'd0, b_take};
  assign b_end      = (b_left_raw == 32'd0);
  assign b_left_n   = b_end ? row_bytes : b_left_raw;
  assign b_row_n    = b_end ? (a_row_n + 32'd1) : a_row_n;
  assign b_rstart_n = b_end ? (a_rstart_n + desc_src_stride) : a_rstart_n;
  assign b_blk_n    = b_end ? ((a_rstart_n + desc_src_stride) & 32'hFFFFFFF0)
                            : (a_blk_n + 32'd16);

  // ---------------------------------------------------------------- 发射条件
  logic issue_v0, issue_v1;
  assign issue_v0 = (state == S_RUN) && (st_row < desc_rows);
  assign issue_v1 = issue_v0 && (a_row_n < desc_rows);

  assign ddr_ren[0]   = issue_v0;
  assign ddr_ren[1]   = issue_v1;
  assign ddr_raddr[0] = st_blk;
  assign ddr_raddr[1] = a_blk_n;

  // ---------------------------------------------------------------- 写命令流水
  logic [AW-1:0] pl_addr [0:LAST];
  logic [3:0]    pl_lo0  [0:LAST];
  logic [3:0]    pl_n0   [0:LAST];
  logic [3:0]    pl_lo1  [0:LAST];
  logic [3:0]    pl_n1   [0:LAST];
  logic          pl_v0   [0:LAST];
  logic          pl_v1   [0:LAST];
  logic          pl_empty;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      for (int i = 0; i <= LAST; i++) begin
        pl_v0[i] <= 1'b0;
        pl_v1[i] <= 1'b0;
      end
    end else begin
      pl_v0[0]   <= issue_v0;
      pl_v1[0]   <= issue_v1;
      pl_addr[0] <= st_wptr;
      pl_lo0[0]  <= {1'b0, a_lo[3:1]};
      pl_n0[0]   <= a_take[4:1];
      pl_lo1[0]  <= {1'b0, b_lo[3:1]};
      pl_n1[0]   <= issue_v1 ? b_n_w : 4'd0;
      for (int i = 1; i <= LAST; i++) begin
        pl_v0[i]   <= pl_v0[i-1];
        pl_v1[i]   <= pl_v1[i-1];
        pl_addr[i] <= pl_addr[i-1];
        pl_lo0[i]  <= pl_lo0[i-1];
        pl_n0[i]   <= pl_n0[i-1];
        pl_lo1[i]  <= pl_lo1[i-1];
        pl_n1[i]   <= pl_n1[i-1];
      end
    end
  end

  always_comb begin
    pl_empty = 1'b1;
    for (int i = 0; i <= LAST; i++)
      if (pl_v0[i] || pl_v1[i]) pl_empty = 1'b0;
  end

  // ---------------------------------------------------------------- SRAM 写侧
  logic [AW-1:0] a_base, b_base;
  assign a_base = pl_addr[LAST];
  assign b_base = pl_addr[LAST] + {{(AW-4){1'b0}}, pl_n0[LAST]};

  always_comb begin
    for (int k = 0; k < NBANK; k++) begin
      wr_valid[k] = 1'b0;
      wr_addr[k]  = '0;
      wr_data[k]  = '0;
    end

    if (pl_v0[LAST] && ddr_rvalid[0]) begin
      for (logic [4:0] w = 5'd0; w < 5'd8; w = w + 5'd1) begin
        if (w >= {1'b0, pl_lo0[LAST]} && w < ({1'b0, pl_lo0[LAST]} + {1'b0, pl_n0[LAST]})) begin
          wr_valid[w] = 1'b1;
          wr_addr[w]  = a_base + {{(AW-5){1'b0}}, (w - {1'b0, pl_lo0[LAST]})};
          wr_data[w]  = ddr_rdata[0][{w[2:0], 4'd0} +: 16];
        end
      end
    end

    if (pl_v1[LAST] && ddr_rvalid[1]) begin
      for (logic [4:0] w = 5'd0; w < 5'd8; w = w + 5'd1) begin
        if (w >= {1'b0, pl_lo1[LAST]} && w < ({1'b0, pl_lo1[LAST]} + {1'b0, pl_n1[LAST]})) begin
          wr_valid[w + 5'd8] = 1'b1;
          wr_addr[w + 5'd8]  = b_base + {{(AW-5){1'b0}}, (w - {1'b0, pl_lo1[LAST]})};
          wr_data[w + 5'd8]  = ddr_rdata[1][{w[2:0], 4'd0} +: 16];
        end
      end
    end
  end

  // ---------------------------------------------------------------- 状态机
  always_ff @(posedge clk) begin
    if (!rst_n) state <= S_IDLE;
    else        state <= next;
  end

  always_comb begin
    next = state;
    case (state)
      S_IDLE : if (desc_start) next = S_CHECK;
      S_CHECK: next = aligned_ok ? S_RUN : S_IDLE;
      S_RUN  : if (!issue_v0 && pl_empty) next = S_DONE;
      S_DONE : next = S_IDLE;
      default: next = S_IDLE;
    endcase
  end

  always_ff @(posedge clk)
    if (rst_n && state == S_CHECK && !aligned_ok)
      $error("dma: sram/ddr 地址必须 16 字节对齐");

  assign busy = (state == S_CHECK) || (state == S_RUN);
  assign done = (state == S_DONE);

  // ---------------------------------------------------------------- 搬运状态更新
  logic [AW-1:0] n0_x, n1_x;
  assign n0_x = {{(AW-4){1'b0}}, a_take[4:1]};
  assign n1_x = {{(AW-4){1'b0}}, issue_v1 ? b_n_w : 4'd0};

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      st_row    <= '0;
      st_rstart <= '0;
      st_blk    <= '0;
      st_left   <= '0;
      st_wptr   <= '0;
    end else if (state == S_CHECK && aligned_ok) begin
      st_row    <= 32'd0;
      st_rstart <= desc_ddr_addr;
      st_blk    <= desc_ddr_addr & 32'hFFFFFFF0;
      st_left   <= row_bytes;
      st_wptr   <= desc_sram_addr[AW:1];
    end else if (issue_v0) begin
      st_row    <= issue_v1 ? b_row_n    : a_row_n;
      st_rstart <= issue_v1 ? b_rstart_n : a_rstart_n;
      st_blk    <= issue_v1 ? b_blk_n    : a_blk_n;
      st_left   <= issue_v1 ? b_left_n   : a_left_n;
      st_wptr   <= st_wptr + n0_x + n1_x;
    end
  end

endmodule
