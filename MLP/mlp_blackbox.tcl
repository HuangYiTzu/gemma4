# =============================================================================
# mlp_blackbox.tcl -- register the mlp_dsp_mac RTL blackbox with Vitis HLS.
#
# mlp_engine calls mlp_dsp_mac() 1536 times per engine (one DSP48E2 each, see
# mlp_dsp_mac.v for why it is RTL).  HLS only knows that function if this
# script ran before csynth_design.  Every flow that SYNTHESISES mlp_engine
# needs it; csim / sw_emu do not (they use the C model in mlp_common.h).
#
#   local Vitis HLS    run_hls.tcl sources this file (csynth / cosim only)
#   v++ -c (hw, hw_emu) pass it as the pre-synthesis Tcl hook of the
#                       mlp_engine kernel:
#                           v++ -c -k mlp_engine ... --hls.pre_tcl <abs>/mlp_blackbox.tcl
#                       or, in a v++ config file (mlp_engine_compile.cfg):
#                           [hls]
#                           pre_tcl=<abs>/mlp_blackbox.tcl
#
# It needs mlp_dsp_mac.v and mlp_dsp_mac.cpp in the SAME directory as this
# script, and it writes mlp_dsp_mac.json there.  The JSON is generated rather
# than checked in because the blackbox wants the RTL and C-model paths, and
# v++ runs HLS in a scratch directory of its own, so relative paths would
# resolve against the wrong place.  Deriving them from this script's own
# location works on both machines without editing anything.
#
# If the mlp_engine synthesis log says the function mlp_dsp_mac is undefined
# (or "unresolved"), this script did not run for that kernel.
# =============================================================================

set mlp_bb_dir [file dirname [file normalize [info script]]]
set mlp_bb_v   [file join $mlp_bb_dir mlp_dsp_mac.v]
set mlp_bb_c   [file join $mlp_bb_dir mlp_dsp_mac.cpp]
set mlp_bb_j   [file join $mlp_bb_dir mlp_dsp_mac.json]

foreach f [list $mlp_bb_v $mlp_bb_c] {
	if {![file exists $f]} {
		error "mlp_blackbox.tcl: $f not found (must sit next to this script)"
	}
}

# latency 4 / II 1: see the timing table at the top of mlp_dsp_mac.v
set fh [open $mlp_bb_j w]
puts $fh "{"
puts $fh "  \"c_function_name\"     : \"mlp_dsp_mac\","
puts $fh "  \"rtl_top_module_name\" : \"mlp_dsp_mac\","
puts $fh "  \"c_files\"             : \[ { \"c_file\" : \"$mlp_bb_c\", \"cflag\" : \"\" } \],"
puts $fh "  \"rtl_files\"           : \[ \"$mlp_bb_v\" \],"
puts $fh "  \"c_parameters\" : \["
puts $fh "    { \"c_name\" : \"w_lo\",  \"c_port_direction\" : \"in\", \"rtl_ports\" : { \"data_read_in\" : \"w_lo\"  } },"
puts $fh "    { \"c_name\" : \"w_hi\",  \"c_port_direction\" : \"in\", \"rtl_ports\" : { \"data_read_in\" : \"w_hi\"  } },"
puts $fh "    { \"c_name\" : \"q\",     \"c_port_direction\" : \"in\", \"rtl_ports\" : { \"data_read_in\" : \"q\"     } },"
puts $fh "    { \"c_name\" : \"first\", \"c_port_direction\" : \"in\", \"rtl_ports\" : { \"data_read_in\" : \"first\" } }"
puts $fh "  \],"
puts $fh "  \"c_return\" : { \"c_port_direction\" : \"out\", \"rtl_ports\" : { \"data_write_out\" : \"ap_return\" } },"
puts $fh "  \"rtl_common_signal\" : {"
puts $fh "    \"module_clock\"        : \"ap_clk\","
puts $fh "    \"module_reset\"        : \"ap_rst\","
puts $fh "    \"module_clock_enable\" : \"ap_ce\","
puts $fh "    \"ap_ctrl_chain_protocol_idle\"     : \"\","
puts $fh "    \"ap_ctrl_chain_protocol_start\"    : \"\","
puts $fh "    \"ap_ctrl_chain_protocol_ready\"    : \"\","
puts $fh "    \"ap_ctrl_chain_protocol_done\"     : \"\","
puts $fh "    \"ap_ctrl_chain_protocol_continue\" : \"\""
puts $fh "  },"
puts $fh "  \"rtl_performance\"    : { \"latency\" : \"4\", \"II\" : \"1\" },"
puts $fh "  \"rtl_resource_usage\" : { \"FF\" : \"2\", \"LUT\" : \"0\", \"BRAM\" : \"0\", \"URAM\" : \"0\", \"DSP\" : \"1\" }"
puts $fh "}"
close $fh

add_files -blackbox $mlp_bb_j
puts "mlp_blackbox.tcl: registered the mlp_dsp_mac RTL blackbox ($mlp_bb_j)"
