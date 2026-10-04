// ============================================================================
// mlp_dsp_mac.v -- ONE packed INT2 x INT8 multiply-accumulate on ONE DSP48E2,
//                  accumulating in the DSP's own P register.
//
// This is the RTL blackbox behind mlp_dsp_mac() (mlp_common.h, blackbox JSON
// written by mlp_blackbox.tcl).  mlp_engine instantiates it 1536 times per
// engine, one per (DSP column d, K lane kl).
//
// WHY RTL.  Written in HLS C++ as  P += a * q , every one of those 1536
// accumulators was built as
//     DSP P out (32 bit) -> fabric copy register + 2:1 "pipeline bypass" mux
//                        -> "first step of the K tile ? 0 : ..." mux
//                        -> back into the DSP's C port
// i.e. 32 LUT + 32 FF + 64 wires per DSP, all selected by one control net
// with 49,152 loads (high_fanout.rpt of the third Hardware build).  HLS
// inserts that bypass mux for every loop-carried variable of a pipelined loop
// and it cannot be written away in C++ (three codings were tried, see
// mlp_engine.cpp).  Here the feedback never leaves the DSP: the Z multiplexer
// of the ALU selects P ("keep accumulating") or 0 ("start a new K tile"),
// switched by the OPMODE pins.
//
//   P <= (first ? 0 : P) + (w_hi * 2^18 + w_lo) * q
//
// Datapath (all inside the DSP48E2):
//   A[29:18] = sign-extended w_hi, A[17:0] = 0      A2 register  (AREG=1)
//   D[26:0]  = sign-extended w_lo                   D  register  (DREG=1)
//   AD = A + D   = w_hi * 2^18 + w_lo  (pre-adder)  AD register  (ADREG=1)
//   B[17:0]  = sign-extended q                      B1, B2       (BREG=2)
//   M  = AD * B                                     M  register  (MREG=1)
//   P  = M + (first ? 0 : P)                        P  register  (PREG=1)
//
// LATENCY 4, II 1.  Inputs are sampled at the clock edge that ends the cycle
// in which HLS presents them (edge 1); A/D/B1 -> edge 1, AD/B2 -> edge 2,
// M -> edge 3, P -> edge 4, so the sum is on ap_return four cycles after
// the call.  The OPMODE register (OPMODEREG=1) is loaded at edge 3 and steers the
// ALU during the cycle that ends at edge 4, so "first" is delayed by two
// fabric registers first.  Those two registers are identical in all 1536
// instances; Vivado merges them into one chain per engine, which is exactly
// one 1536-load net instead of the old 49,152-load one.
//
// NO CLOCK ENABLE, AND THE ONE RULE THE CALLER MUST KEEP.  The unit has no
// way to tell a real call from an idle cycle: HLS refuses an ap_ctrl_chain
// blackbox (the only kind with an ap_start) inside a pipeline (HLS 200-1916),
// so this is ap_ctrl_none and its registers advance on EVERY clock.
// Therefore
//
//     the calls of one K tile must arrive on CONSECUTIVE clocks,
//
// i.e. the loop that calls it must run at II = 1 with no stall and no bubble
// inside a K tile.  Idle clocks BETWEEN tiles are harmless: they add garbage
// to P after the tile's last result has been read (4 clocks after its call)
// and before the next tile's first step overwrites P.  mlp_engine's
// job_ktile_job_kstep loop meets the rule (II = 1, no blocking I/O, so it can
// neither stall nor bubble), and the Hardware-emulation bit-exact check would
// catch a violation.  A test pipeline that fell to II = 13 did fail C/RTL
// co-simulation, which is exactly the failure this rule describes; at II = 1,
// with and without idle clocks between tiles, it passes.
//
// ap_ce is accepted (the blackbox JSON declares it) but deliberately NOT
// used.  In mlp_engine, HLS drives it with the pipeline's FSM state bit; the
// 1536 instances would merge it into one net with ~16 k DSP clock-enable
// loads, and under the rule above it carries no information: it only drops
// after the loop has finished.  The DSP clock enables are tied high instead,
// exactly like the DSP macros HLS generates itself (.ce(1'b1)).
//
// OUTPUT.  Only P[31:0] leaves the DSP.  A K tile adds MLP_KSTEP_HW = 22
// products of |w*q| <= 256 into each packed field, so
//     |P| <= 22 * 256 * (2^18 + 1) = 1.48e9 < 2^31
// (asserted in mlp_common.h).  Every other output of the DSP is unused.
//
// Primitive and attribute names: UltraScale+ DSP48E2 (UG579).
// ============================================================================
`timescale 1 ns / 1 ps

module mlp_dsp_mac (
    input  wire        ap_clk,
    input  wire        ap_rst,
    input  wire        ap_ce,      // unused, see NO CLOCK ENABLE above
    input  wire [1:0]  w_lo,       // INT2, low  packed field (even lane)
    input  wire [1:0]  w_hi,       // INT2, high packed field (odd lane)
    input  wire [7:0]  q,          // INT8 activation shared by the column
    input  wire [0:0]  first,      // 1 on the first step of a K tile
    output wire [31:0] ap_return   // P[31:0] after this step
);

    // ---- "first" -> OPMODE, two stages to line up with edge 3 ------------
    reg first_d1 = 1'b0;
    reg first_d2 = 1'b0;
    always @(posedge ap_clk) begin
        first_d1 <= first[0];
        first_d2 <= first_d1;
    end

    //            W        Z                          Y       X
    //          00       000 = 0 / 010 = P           01      01  = M
    wire [8:0] opmode = {2'b00, (first_d2 ? 3'b000 : 3'b010), 2'b01, 2'b01};

    // ---- operand formatting (wiring only, no logic) -----------------------
    wire [29:0] a_in = {{10{w_hi[1]}}, w_hi, 18'd0};  // w_hi * 2^18
    wire [26:0] d_in = {{25{w_lo[1]}}, w_lo};         // w_lo
    wire [17:0] b_in = {{10{q[7]}}, q};               // q

    wire [47:0] p;
    assign ap_return = p[31:0];

    DSP48E2 #(
        // feature control
        .AMULTSEL            ("AD"),
        .A_INPUT             ("DIRECT"),
        .BMULTSEL            ("B"),
        .B_INPUT             ("DIRECT"),
        .PREADDINSEL         ("A"),
        .RND                 (48'h000000000000),
        .USE_MULT            ("MULTIPLY"),
        .USE_SIMD            ("ONE48"),
        .USE_WIDEXOR         ("FALSE"),
        .XORSIMD             ("XOR24_48_96"),
        // pattern detector: unused
        .AUTORESET_PATDET    ("NO_RESET"),
        .AUTORESET_PRIORITY  ("RESET"),
        .MASK                (48'h3fffffffffff),
        .PATTERN             (48'h000000000000),
        .SEL_MASK            ("MASK"),
        .SEL_PATTERN         ("PATTERN"),
        .USE_PATTERN_DETECT  ("NO_PATDET"),
        // no inverted pins
        .IS_ALUMODE_INVERTED (4'b0000),
        .IS_CARRYIN_INVERTED (1'b0),
        .IS_CLK_INVERTED     (1'b0),
        .IS_INMODE_INVERTED  (5'b00000),
        .IS_OPMODE_INVERTED  (9'b000000000),
        .IS_RSTALLCARRYIN_INVERTED (1'b0),
        .IS_RSTALUMODE_INVERTED    (1'b0),
        .IS_RSTA_INVERTED    (1'b0),
        .IS_RSTB_INVERTED    (1'b0),
        .IS_RSTCTRL_INVERTED (1'b0),
        .IS_RSTC_INVERTED    (1'b0),
        .IS_RSTD_INVERTED    (1'b0),
        .IS_RSTINMODE_INVERTED (1'b0),
        .IS_RSTM_INVERTED    (1'b0),
        .IS_RSTP_INVERTED    (1'b0),
        // pipeline registers (see the latency table above)
        .ACASCREG            (1),
        .ADREG               (1),
        .ALUMODEREG          (0),
        .AREG                (1),
        .BCASCREG            (2),
        .BREG                (2),
        .CARRYINREG          (0),
        .CARRYINSELREG       (0),
        .CREG                (0),
        .DREG                (1),
        .INMODEREG           (0),
        .MREG                (1),
        .OPMODEREG           (1),
        .PREG                (1)
    ) dsp (
        // cascade outputs: unused
        .ACOUT          (),
        .BCOUT          (),
        .CARRYCASCOUT   (),
        .MULTSIGNOUT    (),
        .PCOUT          (),
        // control / status outputs: unused
        .OVERFLOW       (),
        .PATTERNBDETECT (),
        .PATTERNDETECT  (),
        .UNDERFLOW      (),
        // data outputs
        .CARRYOUT       (),
        .P              (p),
        .XOROUT         (),
        // cascade inputs: unused
        .ACIN           (30'd0),
        .BCIN           (18'd0),
        .CARRYCASCIN    (1'b0),
        .MULTSIGNIN     (1'b0),
        .PCIN           (48'd0),
        // control inputs
        .ALUMODE        (4'b0000),          // Z + W + X + Y + CIN
        .CARRYINSEL     (3'b000),
        .CLK            (ap_clk),
        .INMODE         (5'b00100),         // B2, D into the pre-adder, add, A2
        .OPMODE         (opmode),
        // data inputs
        .A              (a_in),
        .B              (b_in),
        .C              (48'hffffffffffff), // unused (Z never selects C)
        .CARRYIN        (1'b0),
        .D              (d_in),
        // clock enables
        .CEA1           (1'b0),             // AREG=1 uses A2 only
        .CEA2           (1'b1),
        .CEAD           (1'b1),
        .CEALUMODE      (1'b0),
        .CEB1           (1'b1),
        .CEB2           (1'b1),
        .CEC            (1'b0),
        .CECARRYIN      (1'b0),
        .CECTRL         (1'b1),            // OPMODE register
        .CED            (1'b1),
        .CEINMODE       (1'b0),
        .CEM            (1'b1),
        .CEP            (1'b1),
        // resets: never used, the first-step OPMODE does the clearing
        .RSTA           (1'b0),
        .RSTALLCARRYIN  (1'b0),
        .RSTALUMODE     (1'b0),
        .RSTB           (1'b0),
        .RSTC           (1'b0),
        .RSTCTRL        (1'b0),
        .RSTD           (1'b0),
        .RSTINMODE      (1'b0),
        .RSTM           (1'b0),
        .RSTP           (1'b0)
    );

endmodule
