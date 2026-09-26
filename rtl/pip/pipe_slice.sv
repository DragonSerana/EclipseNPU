module pipe_slice (
  input  logic       clk,
  input  logic       rst_n,
  input  logic       s_valid,
  input  logic [7:0] s_data,
  output logic       s_ready,
  output logic       m_valid,
  output logic [7:0] m_data,
  input  logic       m_ready 
);
  // 两个寄存器：一个 1 位的标志、一个 8 位的数据
  logic       reg_valid;
  logic [7:0] reg_data;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      reg_valid <= 1'd0;
      reg_data <= 8'd0;
    // 只要reg_valid是空的，我就怎么都能接收数据，ready并不是能不能处理完，而是是否能存储这个clk的数据 
    end else if (s_valid && (!reg_valid || m_ready)) begin
      reg_valid <= s_valid;
      reg_data <= s_data;
    end else if (!s_valid && m_ready && reg_valid) begin
      reg_valid <= 1'd0;
      reg_data <= 8'd0;      
    end
  end

  assign m_valid = reg_valid;
  assign m_data  = reg_data;
  assign s_ready = (m_valid == 1'd0) ? 1'b1 : m_ready ;

endmodule
