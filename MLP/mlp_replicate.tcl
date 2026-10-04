# ===========================================================================
# mlp_replicate.tcl -- mark the few high-fanout registers that broke timing so
#                      that the PLACER replicates them (fix A1, second version).
#
# Sourced at the end of mover_pblocks.tcl, i.e. from the PLACE_DESIGN.TCL.PRE
# hook, on the opt_design'ed netlist just BEFORE place_design.  Grep runme.log
# for "MLP-REPL" to see what it did, and for "Physopt 32-81" inside the
# place_design section to see how many copies the placer made of each net.
# Any error inside is caught and reported: this script can cost the
# experiment, never the build.
#
# ---------------------------------------------------------------------------
# WHAT IT FIXES
#
# The fourth Hardware build routed but failed the timing check on the
# platform's PCIe clock dma_ip_axi_aclk_1 (250 MHz, not scalable) by 0.354 ns,
# and the kernel clock missed by 5.3 ns.  In both cases the worst paths were
# ONE flip-flop driving hundreds of loads spread far apart, 95-99 % wire:
#
#   target          driver (the real names, from the timing reports)    loads
#   dma_hbm_r15     hmss_0/inst/path_12/slice0_12/inst/r15.r_multi/
#                   triple_slr.resp.slr_master/common.srl_fifo_0/
#                   asyncclear_state0_inst                   (19 of 30 paths)  525
#   dma_data_sc     axi_data_sc/inst/s00_entry_pipeline/s00_mmu/inst/
#                   gen_endpoint.r_state_reg[0]              (10 of 30 paths) 1050
#   kernel_q_sign   mlp_engine_N/.../job_ktile_job_kstep_fu_*/
#                   q_*_pp0_iter4_reg_reg[7]    (sign bit of each activation,
#                   24 lanes x 3 engines = 72; every one of the kernel's
#                   worst paths)                                               705
#
# (The 30th DMA path is a single SLR0 -> SLR1 handshake register with fanout
# 1; replication cannot shorten it.  The post-route phys_opt of mlp_link.cfg
# handles that one.)
#
# ---------------------------------------------------------------------------
# WHY THE FIRST VERSION FAILED THE SIXTH BUILD
#
# Version 1 matched nets by broad name patterns ("hmss_0/inst/path_*/slice*",
# "axi_data_sc/inst/*", every q_* bit, every ph_* net) and forced replication
# AFTER placement.  On the real netlist those patterns caught 1,153 nets
# instead of a few dozen -- all 32 HBM slices, SmartConnect resets and
# arbiters, all 8 bits of every q -- and phys_opt added 9,083 registers
# (util_hier of the routed_error dcp: +2,300..2,540 per engine, +1,111
# axi_data_sc, +785 hmss_0) into an already-placed design, right next to the
# DSP columns and the HBM interface.  The router, which had just converged in
# the fourth build, no longer could: 1,390 node overlaps, Constraints 18-1000.
#
# This version differs in four ways:
#   1) EXACT TARGETS.  Each target is a driver-cell pattern that names the
#      whole path (square brackets in Vivado filters are literal), and a
#      match must also be a flip-flop with at least MIN loads.  Every
#      match is printed.
#   2) BEFORE PLACEMENT, BY PROPERTY.  It does not replicate anything itself:
#      it sets FORCE_MAX_FANOUT on the target nets, and place_design's
#      "Physical Synthesis In Placer" phase replicates them WHILE it is
#      still free to move the loads.  Local test (Vivado 2022.1, xcu280,
#      netlist with these exact names): the 525- and 705-load nets were each
#      split into 10 drivers during placement; every other net was left
#      alone.  The copy count is the placer's choice -- the property value
#      is a trigger, not a precise limit (128 and 256 gave the same result),
#      so ~9 new cells per target net, ~700 for all 74, against 9,083 before.
#      Post-place alternatives were tested and rejected:
#      -force_replication_on_nets ignores FORCE_MAX_FANOUT, and setting the
#      property after placement is ignored by phys_opt_design.
#   3) SAFETY CAP.  More than MLP_REP_MAX_NETS matches means the patterns hit
#      something unintended: the script then sets NOTHING and says so.
#   4) DRY RUN.  "set MLP_REP_DRYRUN 1" before sourcing it only prints what it
#      would mark.  Use it on the fourth build's routed checkpoint to check
#      the match list on the real netlist before spending a 5-hour build:
#          open_checkpoint level0_wrapper_routed.dcp
#          set MLP_REP_DRYRUN 1
#          source <abs path>/mlp_replicate.tcl
#      (expect 1 + 1 + 72 = 74 nets).
#
# NINTH BUILD.  mlp_engine is re-synthesised for the 4-lane link (kernel.h,
# FOUR LANES PER ENGINE).  Only its load loop changed, not job_kstep, so the
# q_*_pp0_iter4_reg_reg[7] names should come out the same -- but check that
# runme.log still says "kernel_q_sign 72 nets".  If it says 0, only the
# kernel-clock part of A1 is lost (that clock is scaled automatically); the
# two DMA targets are platform names and do not depend on the kernels.
#
# Settings that may be overridden by setting the variable before sourcing:
#   MLP_REP_DRYRUN    0    1 = report only, change nothing
#   MLP_REP_MAX_NETS  100  abort (change nothing) above this many target nets
#   MLP_REP_LIMIT     128  FORCE_MAX_FANOUT value written on each target net
#   MLP_REP_DISABLE   0    1 = skip this script entirely
# ===========================================================================

if {![info exists MLP_REP_DRYRUN]}   { set MLP_REP_DRYRUN   0 }
if {![info exists MLP_REP_MAX_NETS]} { set MLP_REP_MAX_NETS 100 }
if {![info exists MLP_REP_LIMIT]}    { set MLP_REP_LIMIT    128 }
if {![info exists MLP_REP_DISABLE]}  { set MLP_REP_DISABLE  0 }

# label, driver-cell pattern (full hierarchical name, Vivado glob -- '*' may
# span '/', '[' ']' are literal), minimum loads, expected number of drivers
set MLP_REP_TARGETS {
	{dma_hbm_r15
	 "*/hmss_0/inst/path_12/slice0_12/inst/r15.r_multi/triple_slr.resp.slr_master/common.srl_fifo_0/asyncclear_state0_inst"
	 256 1}
	{dma_data_sc
	 "*/axi_data_sc/inst/s00_entry_pipeline/s00_mmu/inst/gen_endpoint.r_state_reg[0]"
	 256 1}
	{kernel_q_sign
	 "*/mlp_engine_*/inst/grp_compute_job_fu_*/grp_compute_job_Pipeline_job_ktile_job_kstep_fu_*/q_*_pp0_iter4_reg_reg[7]"
	 256 72}
}

set MLP_REP_VERSION "2026-09-29 (A1 v2, unchanged targets)"

puts "MLP-REPL: ==================================================="
puts "MLP-REPL: mlp_replicate.tcl version $MLP_REP_VERSION"

# The net driven by a register's Q pin, and its load count ("" / -1 if none)
proc mlp_rep_q_net {cell} {
	set q [get_pins -quiet -of_objects $cell -filter {REF_PIN_NAME == Q}]
	if {[llength $q] != 1} { return [list "" -1] }
	set n [get_nets -quiet -of_objects $q]
	if {[llength $n] != 1} { return [list "" -1] }
	return [list $n [expr {[get_property FLAT_PIN_COUNT $n] - 1}]]
}

# Collect {label cell net loads} for every valid target driver
proc mlp_rep_collect {} {
	global MLP_REP_TARGETS
	set out {}
	foreach t $MLP_REP_TARGETS {
		lassign $t label pat minload expect
		set cells [get_cells -quiet -hierarchical -filter "NAME =~ \"$pat\""]
		set n_ok 0; set n_small 0; set n_notff 0
		foreach c $cells {
			if {![string match "FD*" [get_property REF_NAME $c]]} { incr n_notff; continue }
			lassign [mlp_rep_q_net $c] net loads
			if {$net eq "" || $loads < $minload} {
				incr n_small
				puts "MLP-REPL:   skip  $c ($loads loads < $minload)"
				continue
			}
			lappend out [list $label $c $net $loads]
			incr n_ok
		}
		set note ""
		if {$n_ok != $expect} { set note "   <-- expected $expect, CHECK THE NAMES" }
		puts [format "MLP-REPL: %-14s %3d nets  (%d below the load minimum, %d not a flip-flop)%s" \
		      $label $n_ok $n_small $n_notff $note]
	}
	return $out
}

if {$MLP_REP_DISABLE} {
	puts "MLP-REPL: disabled (MLP_REP_DISABLE = 1), nothing marked"
} elseif {[catch {
	set tg [mlp_rep_collect]
	set n  [llength $tg]
	set maxl 0; set suml 0
	foreach e $tg {
		set l [lindex $e 3]
		incr suml $l
		if {$l > $maxl} { set maxl $l }
	}
	puts "MLP-REPL: $n target nets, $suml loads in total, largest $maxl"

	if {$n == 0} {
		puts "MLP-REPL: WARNING - no target found, nothing marked."
	} elseif {$n > $MLP_REP_MAX_NETS} {
		puts "MLP-REPL: ERROR - $n target nets is more than the safety cap"
		puts "MLP-REPL:         MLP_REP_MAX_NETS = $MLP_REP_MAX_NETS: the patterns match"
		puts "MLP-REPL:         something unintended.  NOTHING was marked; the"
		puts "MLP-REPL:         build continues without replication."
		foreach e [lrange $tg 0 9] { puts "MLP-REPL:   e.g. [lindex $e 1] ([lindex $e 3] loads)" }
	} else {
		foreach e $tg {
			lassign $e label c net loads
			puts [format "MLP-REPL:   %-14s %5d loads  %s" $label $loads $c]
		}
		if {$MLP_REP_DRYRUN} {
			puts "MLP-REPL: DRY RUN (MLP_REP_DRYRUN = 1): nothing marked"
		} else {
			set nets {}
			foreach e $tg { lappend nets [lindex $e 2] }
			set_property FORCE_MAX_FANOUT $MLP_REP_LIMIT [get_nets $nets]
			puts "MLP-REPL: FORCE_MAX_FANOUT = $MLP_REP_LIMIT set on $n nets;"
			puts "MLP-REPL: place_design replicates them (see \"Physopt 32-81\" in its log)"
		}
	}
} msg]} {
	puts "MLP-REPL: ERROR - $msg -- nothing marked, the run continues"
}
puts "MLP-REPL: ==================================================="
