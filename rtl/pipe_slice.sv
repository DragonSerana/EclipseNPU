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
  logic [1:0] cnt;
  logic [7:0] r0;
  logic [7:0] r1;
  logic push, pop;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      cnt <= 2'd0;
      r0 <= 8'd0;
      r1 <= 8'd0;
    end else begin
      cnt <= cnt+push-pop;
      if (pop && cnt == 2'd2)     r0 <= r1;
      else if (push && (cnt==0 || pop)) r0 <= s_data;
      
      if (cnt==1 && push && !pop) r1 <= s_data;
    end
  end

  assign m_valid = cnt > 0;
  assign m_data  = r0;
  assign s_ready = cnt < 2;
  assign push = s_valid && s_ready;
  assign pop = m_valid && m_ready;

endmodule
