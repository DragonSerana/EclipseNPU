// R1 之前的工具链热身：8 位计数器。
// 目的只有一个——学会 Verilator 的节奏（一个 posedge = cmodel 里的一拍）。
module counter (
  input  logic       clk,
  input  logic       rst_n,
  input  logic       en,
  output logic [7:0] cnt,
  output logic [7:0] cnt_next
);
  always_ff @(posedge clk) begin
    if (!rst_n)  cnt <= 8'd0;
    else if (en) cnt <= cnt + 8'd1;
    // else if (en) cnt <= cnt_next;
  end

  assign cnt_next = cnt + 8'd1;
endmodule
