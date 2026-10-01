module sram (
  input  logic       clk,
  input  logic        we,   
  input  logic        rst_n,         
  input  logic [9:0]  waddr,    
  input  logic [15:0] wdata,    

  input  logic [9:0]  raddr,    
  output logic [15:0] rdata    
);
  logic [15:0] mem [0:1023];

  always_ff @(posedge clk) begin
    if (!rst_n) rdata <= 16'd0;
    else        rdata <= mem[raddr];
  end

  always_ff @(posedge clk) begin
    if (we) mem[waddr] <= wdata;
  end

endmodule
