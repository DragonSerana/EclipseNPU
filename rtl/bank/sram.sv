
module sram #(
  parameter DEPTH = 1024,   
  parameter DW    = 16      
) (
  input  logic                     clk,
  input  logic                     rst_n,

  input  logic                     we,
  input  logic [$clog2(DEPTH)-1:0] waddr,
  input  logic [DW-1:0]            wdata,
  input  logic [$clog2(DEPTH)-1:0] raddr,
  output logic [DW-1:0]            rdata
);
  logic [DW-1:0] mem[0:DEPTH-1];

  always_ff @(posedge clk) begin
    if (!rst_n) rdata <= {DW{1'b0}};
    else        rdata <= mem[raddr];
  end

  always_ff @(posedge clk) begin
    if (we) mem[waddr] <= wdata;
  end

endmodule
