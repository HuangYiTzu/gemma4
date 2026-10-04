/*******************************************************************************
** HOST code for the Gemma-4-E2B MLP accelerator (Alveo U280, Vitis / OpenCL)
**
**   o) detects the Xilinx platform / target device, loads the xclbin
**   o) loads the real INT2 QAT weights and scales of layer 15 from the
**      bin_packing_mlp.py export (mlp_bin_output/layer_15; format at the top
**      of mlp_model.h).  The export already has the HBM layout:
**        - W_gate / W_up : fused column zip, ^0xAA, 512-column macro tiles,
**                          SLR(t) = t/16, 16 columns per pseudo-channel
**        - W_down        : K split, SLR(n_tile,k_tile) = k_tile/8,
**                          4 rows per pseudo-channel
**      so gate_up_weight_pcNN.bin ++ down_weight_slr{0,1,2}_pcNN.bin is
**      channel image NN as is.
**   o) derives the per-channel requant multipliers r_g / r_u / c_down from
**      the checkpoint's weight and activation scales
**   o) pins every channel image to its own HBM bank (XCL_MEM_TOPOLOGY), which
**      is what lets the SLR0 data movers own all 32 AXI ports locally
**   o) runs the accelerator and checks the 1536-wide INT16 down_proj codes
**      bit-exactly against the golden model (the datapath is all integer)
**
** The accelerator is NINE compute units (mlp_link.cfg):
**     ctrl_1     (mlp_ctrl,    SLR0)      x_q / r_g / r_u -> 3 control lanes
**     mover_1..4 (mlp_mover,   SLR0)      8 HBM ports each; CU j owns
**                                         HBM[8j..8j+7] and engine lane j
**     eng_1/2/3  (mlp_engine,  SLR0/1/2)  one MAC array per SLR, FREE-RUNNING
**     collect_1  (mlp_collect, SLR0)      c_down + y_q
** Only the six SLR0 kernels have arguments and are enqueued here; the engines
** have stream ports only (ap_ctrl_none) and are started by the data that the
** control kernel and the movers push, so the host never touches them.
**
** WHY FOUR MOVERS.  One 32-port mover made every SLR-crossing net start from
** the same block, and SLLs are allocated per column: route_design measured
** 207 % demand on one column against 14 % on another and refused the design
** (VPL 35-3, global congestion level 7) although the boundary total was only
** 83 %.  Four CUs anchored to four contiguous HBM blocks start their crossings
** from four points across the die width.  See kernel.h for the full argument.
** Nothing about the DATA changed -- the weight files and their HBM banks are
** exactly as before, only which CU reads which bank.
**
** The four movers are four CUs of ONE kernel, so they are selected by CU name:
**     clCreateKernel(program, "mlp_mover:{mlp_mover_1}", ...)
**
** The v++ link must place the ports on matching banks, e.g.
**     [connectivity]
**     sp=mlp_mover_1.w_hbm_0:HBM[0]  ... sp=mlp_mover_1.w_hbm_7:HBM[7]
**     sp=mlp_mover_2.w_hbm_0:HBM[8]  ... and so on to mlp_mover_4 / HBM[31]
**     sp=mlp_ctrl_1.x_q:PLRAM[0]      (x_q / r_g / r_u = gmemS bundle)
**     sp=mlp_collect_1.c_down:PLRAM[1] (c_down / y_q    = gmemC bundle)
**                                 NOT on HBM -- hmss_0 has only 33 slots;
**                                 NOT on DDR[0] -- that instantiates a whole
**                                 DDR4 controller in the busiest SLR
**     slr=mlp_engine_2:SLR1
** The control-plane buffers below are created WITHOUT a bank flag on
** purpose: XRT allocates each one in whatever bank its kernel argument is
** connected to, so moving them between DDR / PLRAM is a link-only change.
**
** Based on the Xilinx 2018 OpenCL host template.
*******************************************************************************/

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <cstring>
#include <cmath>
#include <iostream>
#include <string>
#include <fstream>
#include <iomanip>

using namespace std;

#include <CL/cl.h>
#include <CL/cl_ext.h>

#include "help_functions.h"

#define MLP_HOST_ONLY          // plain C++ view of kernel.h (no ap_fixed)
#include "kernel.h"
#include "mlp_model.h"

#define ALL_MESSAGES

#ifndef XCL_MEM_TOPOLOGY
#define XCL_MEM_TOPOLOGY (1 << 31)
#endif

// ---------------------------------------------------------------------------
// Pass/fail: the kernel output is an INT16 code and every stage from x_q to
// y_q is integer arithmetic, so the golden model predicts it EXACTLY.  Any
// nonzero difference is a real error (layout, DSP split, transfer, ...).
// ---------------------------------------------------------------------------

// kernel arguments: a mover CU takes its MLP_NPC_MV HBM channels, the control
// kernel takes x_q / r_g / r_u and the collector c_down / y_q.  The three
// engines have none (stream ports only, ap_ctrl_none).
#define MLP_NB_ARGS_MOVER   MLP_NPC_MV
#define MLP_NB_ARGS_CTRL    3
#define MLP_NB_ARGS_COLLECT 2

// enqueued kernels per test case: the collector, the control kernel and the
// MLP_NMV movers.  Their events sit at K_exe_event[c * MLP_KPC + ...] in the
// order below; the result read-back only has to wait on the collector, which
// is the one that writes y_q.
#define MLP_KPC             (2 + MLP_NMV)
#define MLP_KEV_COLLECT(c)  ((c) * MLP_KPC + 0)
#define MLP_KEV_CTRL(c)     ((c) * MLP_KPC + 1)
#define MLP_KEV_MOVER(c, j) ((c) * MLP_KPC + 2 + (j))

static void *aligned_alloc_or_die(size_t bytes, const char *what)
{
	void *ptr = nullptr;
	if (posix_memalign(&ptr, 4096, bytes)) {
		cout << endl << "HOST-Error: Out of Memory allocating " << what
		     << " (" << bytes << " B)" << endl << endl;
		exit(EXIT_FAILURE);
	}
	return ptr;
}

// ********************************************************************************** //
// ---------------------------------------------------------------------------------- //
//                          M A I N    F U N C T I O N                                 //
// ---------------------------------------------------------------------------------- //
// ********************************************************************************** //

int main(int argc, char* argv[])
{
	cout << endl;

	// ============================================================================
	// Step 1: Check Command Line Arguments
	// ============================================================================
	#ifdef ALL_MESSAGES
	cout << "HOST-Info: ============================================================= " << endl;
	cout << "HOST-Info: (Step 1) Check Command Line Arguments                      " << endl;
	cout << "HOST-Info: ============================================================= " << endl;
	#endif

	if (argc < 4 || argc > 6)
	{
		cout << "HOST-Error: Incorrect command line syntax " << endl;
		cout << "HOST-Info:  Usage: " << argv[0] << " <Platform_Vendor> <Device_Name> <XCLBIN_File> [Nb_Of_Test_Cases] [Layer_Dir]" << endl << endl;
		return EXIT_FAILURE;
	}

	const char* Target_Platform_Vendor   = argv[1];
	const char* Target_Device_Name       = argv[2];
	const char* xclbinFilename           = argv[3];

	// One test case = one activation vector x_q (mlp_model.h, MLP_TC_*).
	// The weights and scales are NOT part of a case: they come from the
	// checkpoint and stay resident in HBM, exactly as at decode time where
	// only the activation changes per token.  sw_emu is slow, so the count is
	// settable from the command line.
	const int NB_CASES = (argc >= 5) ? atoi(argv[4]) : MLP_NB_TESTS;
	if (NB_CASES < 1) {
		cout << "HOST-Error: Nb_Of_Test_Cases must be >= 1" << endl << endl;
		return EXIT_FAILURE;
	}
	char layer_dirbuf[1024];
	const char *layer_dir = (argc == 6) ? argv[5]
	                      : mlp_find_layer_dir(layer_dirbuf, sizeof layer_dirbuf);

	cout << "HOST-Info: Platform_Vendor   : " << Target_Platform_Vendor << endl;
	cout << "HOST-Info: Device_Name       : " << Target_Device_Name << endl;
	cout << "HOST-Info: XCLBIN_file       : " << xclbinFilename << endl;
	cout << "HOST-Info: Model             : Gemma-4-E2B MLP layer " << MLP_LAYER_IDX
	     << ", K=" << MLP_K << " F=" << MLP_F << ", INT2 weights, decode stage" << endl;
	cout << "HOST-Info: Layer export      : " << layer_dir << endl;
	cout << "HOST-Info: Test cases        : " << NB_CASES
	     << " (weights loaded once, activations refreshed per case)" << endl;

	// ============================================================================
	// Step 2: Detect Target Platform and Target Device in a system.
	//         Create Context and Command Queue.
	// ============================================================================
	cout << endl;
	#ifdef ALL_MESSAGES
	cout << "HOST-Info: ============================================================= " << endl;
	cout << "HOST-Info: (Step 2) Detect Target Platform and Target Device in a system " << endl;
	cout << "HOST-Info:          Create Context and Command Queue                     " << endl;
	cout << "HOST-Info: ============================================================= " << endl;
	#endif

	cl_uint         ui;

	cl_platform_id      *Platform_IDs;
	cl_uint             Nb_Of_Platforms;
	cl_platform_id      Target_Platform_ID;
	bool                Platform_Detected;
	char                *platform_info;

	cl_device_id        *Device_IDs;
	cl_uint             Nb_Of_Devices;
	cl_device_id        Target_Device_ID;
	bool                Device_Detected;
	char                *device_info;

	cl_context          Context;
	cl_command_queue    Command_Queue;

	cl_int              errCode;
	size_t              size;

	// ------------------------------------------------------------------------------------
	// Step 2.1: Get All PLATFORMS, then search for Target_Platform_Vendor (CL_PLATFORM_VENDOR)
	// ------------------------------------------------------------------------------------
	errCode = clGetPlatformIDs(0, NULL, &Nb_Of_Platforms);
	if (errCode != CL_SUCCESS || Nb_Of_Platforms <= 0) {
		cout << endl << "HOST-Error: Failed to get the number of available platforms" << endl << endl;
		return EXIT_FAILURE;
	}

	#ifdef ALL_MESSAGES
	cout << "HOST-Info: Number of detected platforms : " << Nb_Of_Platforms << endl;
	#endif

	Platform_IDs = new cl_platform_id[Nb_Of_Platforms];
	if (!Platform_IDs) {
		cout << endl << "HOST-Error: Out of Memory during memory allocation for Platform_IDs" << endl << endl;
		return EXIT_FAILURE;
	}

	errCode = clGetPlatformIDs(Nb_Of_Platforms, Platform_IDs, NULL);
	if (errCode != CL_SUCCESS) {
		cout << endl << "HOST-Error: Failed to get the available platforms" << endl << endl;
		return EXIT_FAILURE;
	}

	Platform_Detected = false;
	for (ui = 0; ui < Nb_Of_Platforms; ui++) {

		errCode = clGetPlatformInfo(Platform_IDs[ui], CL_PLATFORM_VENDOR, 0, NULL, &size);
		if (errCode != CL_SUCCESS) {
			cout << endl << "HOST-Error: Failed to get the size of the Platform parameter " << "CL_PLATFORM_VENDOR" << " value " << endl << endl;
			return EXIT_FAILURE;
		}

		platform_info = new char[size];
		if (!platform_info) {
			cout << endl << "HOST-Error: Out of Memory during memory allocation for Platform Parameter " << "CL_PLATFORM_VENDOR" << endl << endl;
			return EXIT_FAILURE;
		}

		errCode = clGetPlatformInfo(Platform_IDs[ui], CL_PLATFORM_VENDOR, size, platform_info , NULL);
		if (errCode != CL_SUCCESS) {
			cout << endl << "HOST-Error: Failed to get the " << "CL_PLATFORM_VENDOR" << " platform info" << endl << endl;
			return EXIT_FAILURE;
		}

		if (strcmp(platform_info, Target_Platform_Vendor) == 0) {
			Platform_Detected        = true;
			Target_Platform_ID       = Platform_IDs[ui];
			#ifdef ALL_MESSAGES
			cout << "HOST-Info: Selected platform            : " << Target_Platform_Vendor << endl << endl;
			#endif
		}
	}

	if (Platform_Detected == false) {
		cout << endl << "HOST-Error: Failed to get detect " << Target_Platform_Vendor << " platform" << endl << endl;
		return EXIT_FAILURE;
	}

	// ------------------------------------------------------------------------------------
	// Step 2.2:  Get All Devices for selected platform Target_Platform_ID
	//            then search for CL_DEVICE_NAME = Target_Device_Name
	// ------------------------------------------------------------------------------------
	errCode = clGetDeviceIDs(Target_Platform_ID, CL_DEVICE_TYPE_ALL, 0, NULL, &Nb_Of_Devices);
	if (errCode != CL_SUCCESS) {
		cout << endl << "HOST-Error: Failed to get the number of available Devices" << endl << endl;
		return EXIT_FAILURE;
	}
	#ifdef ALL_MESSAGES
	cout << "HOST-Info: Number of available devices  : " << Nb_Of_Devices << endl;
	#endif

	Device_IDs = new cl_device_id[Nb_Of_Devices];
	if (!Device_IDs) {
		cout << endl << "HOST-Error: Out of Memory during memory allocation for Device_IDs" << endl << endl;
		return EXIT_FAILURE;
	}

	errCode = clGetDeviceIDs(Target_Platform_ID, CL_DEVICE_TYPE_ALL, Nb_Of_Devices, Device_IDs, NULL);
	if (errCode != CL_SUCCESS) {
		cout << endl << "HOST-Error: Failed to get available Devices" << endl << endl;
		return EXIT_FAILURE;
	}

	Device_Detected = false;
	for (ui = 0; ui < Nb_Of_Devices; ui++) {
		errCode = clGetDeviceInfo(Device_IDs[ui], CL_DEVICE_NAME, 0, NULL, &size);
		if (errCode != CL_SUCCESS) {
			cout << endl << "HOST-Error: Failed to get the size of the Device parameter value " << "CL_DEVICE_NAME" << endl << endl;
			return EXIT_FAILURE;
		}

		device_info = new char[size];
		if (!device_info) {
			cout << endl << "HOST-Error: Out of Memory during memory allocation for Device parameter " << "CL_DEVICE_NAME" << " value " << endl << endl;
			return EXIT_FAILURE;
		}

		errCode = clGetDeviceInfo(Device_IDs[ui], CL_DEVICE_NAME, size, device_info, NULL);
		if (errCode != CL_SUCCESS) {
			cout << endl << "HOST-Error: Failed to get the " << "CL_DEVICE_NAME" << " device info" << endl << endl;
			return EXIT_FAILURE;
		}

		if (strcmp(device_info, Target_Device_Name) == 0) {
			Device_Detected        = true;
			Target_Device_ID       = Device_IDs[ui];
		}
	}

	if (Device_Detected == false) {
		cout << endl << "HOST-Error: Failed to get detect " << Target_Device_Name << " device" << endl << endl;
		return EXIT_FAILURE;
	} else {
		#ifdef ALL_MESSAGES
		cout << "HOST-Info: Selected device              : " << Target_Device_Name << endl << endl;
		#endif
	}

	// ------------------------------------------------------------------------------------
	// Step 2.3: Create Context
	// ------------------------------------------------------------------------------------
	#ifdef ALL_MESSAGES
	cout << "HOST-Info: Creating Context ... " << endl;
	#endif
	Context = clCreateContext(0, 1, &Target_Device_ID, NULL, NULL, &errCode);
	if (errCode != CL_SUCCESS) {
		cout << endl << "HOST-Error: Failed to create a Context" << endl << endl;
		return EXIT_FAILURE;
	}

	// ------------------------------------------------------------------------------------
	// Step 2.4: Create Command Queue
	// ------------------------------------------------------------------------------------
	#ifdef ALL_MESSAGES
	cout << "HOST-Info: Creating Command Queue ... " << endl;
	#endif
	Command_Queue = clCreateCommandQueue(Context, Target_Device_ID, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE, &errCode);
	if (errCode != CL_SUCCESS) {
		cout << endl << "HOST-Error: Failed to create a Command Queue" << endl << endl;
		return EXIT_FAILURE;
	}

	// ============================================================================
	// Step 3: Create Program and Kernel
	// ============================================================================
	#ifdef ALL_MESSAGES
	cout << endl;
	cout << "HOST-Info: ============================================================= " << endl;
	cout << "HOST-Info: (Step 3) Create Program and Kernels                           " << endl;
	cout << "HOST-Info: ============================================================= " << endl;
	#endif

	unsigned char *xclbin_Memory;
	int program_length;

	#ifdef ALL_MESSAGES
	cout << "HOST-Info: Loading " << xclbinFilename << " binary file to memory ..." << endl;
	#endif

	program_length = loadFile2Memory(xclbinFilename, (char **) &xclbin_Memory);
	if (program_length < 0) {
		cout << endl << "HOST-Error: Failed to load " << xclbinFilename << " binary file to memory" << endl << endl;
		return EXIT_FAILURE;
	}

	size_t     Program_Length_in_Bytes;
	cl_program Program;
	cl_int     Binary_Status;

	#ifdef ALL_MESSAGES
	cout << "HOST-Info: Creating Program with Binary ..." << endl;
	#endif
	Program_Length_in_Bytes = program_length;
	Program = clCreateProgramWithBinary(Context, 1, &Target_Device_ID, &Program_Length_in_Bytes,
	                                    (const unsigned char **) &xclbin_Memory, &Binary_Status, &errCode);
	if (errCode != CL_SUCCESS) {
		cout << endl << "HOST-Error: Failed to create a Program from a Binary" << endl << endl;
		return EXIT_FAILURE;
	}

	#ifdef ALL_MESSAGES
	cout << "HOST-Info: Building the Program ..." << endl;
	#endif

	errCode = clBuildProgram(Program, 1, &Target_Device_ID, NULL, NULL, NULL);
	if (errCode != CL_SUCCESS) {
		cout << endl << "HOST-Error: Failed to build a Program Executable" << endl << endl;
		return EXIT_FAILURE;
	}

	cl_kernel K_ctrl, K_mover[MLP_NMV], K_collect;

	#ifdef ALL_MESSAGES
	cout << "HOST-Info: Creating the Kernels: mlp_ctrl, mlp_mover x" << MLP_NMV
	     << ", mlp_collect ..." << endl;
	cout << "HOST-Info:   (mlp_engine x3 are free-running: no arguments, never enqueued)" << endl;
	#endif
	K_ctrl = clCreateKernel(Program, "mlp_ctrl", &errCode);
	if (errCode != CL_SUCCESS) {
		cout << endl << "HOST-Error: Failed to create K_ctrl" << endl << endl;
		return EXIT_FAILURE;
	}

	// The MLP_NMV movers are compute units of ONE kernel, so each has to be
	// requested by CU name -- "mlp_mover" alone would bind to whichever CU XRT
	// picks and the other three would never start.  CU j reads HBM[8j..8j+7]
	// (mlp_link.cfg), which is what fixes the host-side argument split below.
	for (int j = 0; j < MLP_NMV; ++j) {
		char kname[64];
		snprintf(kname, sizeof kname, "mlp_mover:{mlp_mover_%d}", j + 1);
		K_mover[j] = clCreateKernel(Program, kname, &errCode);
		if (errCode != CL_SUCCESS) {
			cout << endl << "HOST-Error: Failed to create " << kname << endl << endl;
			return EXIT_FAILURE;
		}
	}

	K_collect = clCreateKernel(Program, "mlp_collect", &errCode);
	if (errCode != CL_SUCCESS) {
		cout << endl << "HOST-Error: Failed to create K_collect" << endl << endl;
		return EXIT_FAILURE;
	}

	// ================================================================
	// Step 4: Prepare Data to Run Kernel
	// ================================================================
	#ifdef ALL_MESSAGES
	cout << endl;
	cout << "HOST-Info: ============================================================= " << endl;
	cout << "HOST-Info: (Step 4) Prepare Data to Run Kernel                           " << endl;
	cout << "HOST-Info: ============================================================= " << endl;
	#endif

	// ------------------------------------------------------------------
	// Step 4.1: Allocate the HBM channel images and the control vectors
	// ------------------------------------------------------------------
	unsigned char *ch_img[MLP_NPC];
	for (int p = 0; p < MLP_NPC; ++p)
		ch_img[p] = (unsigned char *)aligned_alloc_or_die(MLP_CH_BYTES, "HBM channel image");

	// x_q / r_g / r_u stay plain INT32 arrays here, but the kernel reads them
	// as 512-bit words (kernel.h): one word = 16 contiguous INT32, which is
	// exactly what a little-endian AXI master sees.  That needs the buffers
	// 64 B aligned and a whole number of words long -- posix_memalign(4096)
	// covers the first, and 1536 / 12288 INT32 are 96 / 768 words exactly.
	int *x_q      = (int *)aligned_alloc_or_die(MLP_K        * sizeof(int), "x_q");
	int *r_g      = (int *)aligned_alloc_or_die(MLP_F        * sizeof(int), "r_g");
	int *r_u      = (int *)aligned_alloc_or_die(MLP_F        * sizeof(int), "r_u");
	int *c_down   = (int *)aligned_alloc_or_die(MLP_N_DOWN   * sizeof(int), "c_down");
	int *RES      = (int *)aligned_alloc_or_die(MLP_N_DOWN   * sizeof(int), "RES");
	memset(RES, 0, MLP_N_DOWN * sizeof(int));

	// per-case golden outputs, hardware outputs and coverage counters.  The
	// golden model is run for EVERY case up front so that no host-side compute
	// sits between two kernel launches and pollutes the profiling window.
	int       **gold_y = new int    *[NB_CASES];   // golden INT16 codes
	int       **hw_y   = new int    *[NB_CASES];   // kernel INT16 codes
	double    **ref_y  = new double *[NB_CASES];   // real p * ws_down * S_H
	mlp_cov_t  *cov    = new mlp_cov_t[NB_CASES];
	for (int c = 0; c < NB_CASES; ++c) {
		gold_y[c] = new int[MLP_N_DOWN];
		hw_y[c]   = new int[MLP_N_DOWN];
		ref_y[c]  = new double[MLP_N_DOWN];
	}

	// mlp_load_layer fills one contiguous [32][MLP_CH_BYTES] block (the weight
	// files are the channel images), which is then split across the 32
	// page-aligned per-channel allocations that back the HBM buffers.
	static mlp_layer_t layer;
	cout << "HOST-Info: Loading layer " << MLP_LAYER_IDX << " weights / scales into "
	     << MLP_NPC << " HBM channel images ("
	     << MLP_NPC * (long)MLP_CH_BYTES / (1024 * 1024) << " MB) ... ";
	cout.flush();
	{
		static mlp_img_t staging;
		if (mlp_load_layer(layer_dir, staging, &layer)) {
			cout << endl << "HOST-Error: cannot load the layer export from "
			     << layer_dir << endl;
			return EXIT_FAILURE;
		}
		cout << "done" << endl;
		if (mlp_check_packing(staging, &layer)) {
			cout << "HOST-Error: HBM packing contract self-check FAILED" << endl;
			return EXIT_FAILURE;
		}
		cout << "HOST-Info: packing contract self-check  : ok" << endl;
		for (int p = 0; p < MLP_NPC; ++p)
			memcpy(ch_img[p], staging[p], MLP_CH_BYTES);
	}

	// requant multipliers: from the checkpoint scales, once per layer
	if (mlp_layer_requant(&layer, r_g, r_u, c_down)) {
		cout << "HOST-Error: a requant multiplier does not fit INT32" << endl;
		return EXIT_FAILURE;
	}
	cout << "HOST-Info: S_IN " << MLP_S_IN << "  S_G " << MLP_S_G << "  S_U " << MLP_S_U
	     << "  S_H " << MLP_S_H << "  S_Y " << MLP_S_Y
	     << "  (G' Q" << 16 - MLP_GLUT_FRAC << "." << MLP_GLUT_FRAC << ")" << endl;

	// ------------------------------------------------------------------
	// Step 4.2: Create Buffers in Global Memory
	//   Every weight image is pinned to its own HBM pseudo-channel so that the
	//   SLR0 data movers reach all 32 AXI ports locally (zero SLL).  The bank
	//   assignment is unchanged by the four-mover split -- image p still lives
	//   in HBM[p]; all that changed is which of the four CUs reads it.
	// ------------------------------------------------------------------
	#ifdef ALL_MESSAGES
	cout << "HOST-Info: Allocating buffers in Global Memory ..." << endl;
	#endif

	cl_mem GlobMem_W[MLP_NPC];
	cl_mem_ext_ptr_t ext_W[MLP_NPC];

	for (int p = 0; p < MLP_NPC; ++p) {
		memset(&ext_W[p], 0, sizeof(cl_mem_ext_ptr_t));
		ext_W[p].obj   = ch_img[p];
		ext_W[p].param = 0;
		ext_W[p].flags = XCL_MEM_TOPOLOGY | (unsigned)p;      // HBM[p]
		GlobMem_W[p] = clCreateBuffer(Context,
		                CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
		                MLP_CH_BYTES, &ext_W[p], &errCode);
		if (errCode != CL_SUCCESS) {
			cout << endl << "Host-Error: Failed to allocate HBM buffer " << p << endl << endl;
			return EXIT_FAILURE;
		}
	}

	cl_mem GlobMem_x   = clCreateBuffer(Context, CL_MEM_READ_ONLY  | CL_MEM_USE_HOST_PTR, MLP_K      * sizeof(int), x_q,      &errCode);
	cl_mem GlobMem_rg  = clCreateBuffer(Context, CL_MEM_READ_ONLY  | CL_MEM_USE_HOST_PTR, MLP_F      * sizeof(int), r_g,      &errCode);
	cl_mem GlobMem_ru  = clCreateBuffer(Context, CL_MEM_READ_ONLY  | CL_MEM_USE_HOST_PTR, MLP_F      * sizeof(int), r_u,      &errCode);
	cl_mem GlobMem_cd  = clCreateBuffer(Context, CL_MEM_READ_ONLY  | CL_MEM_USE_HOST_PTR, MLP_N_DOWN * sizeof(int), c_down,   &errCode);
	cl_mem GlobMem_RES = clCreateBuffer(Context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, MLP_N_DOWN * sizeof(int), RES,      &errCode);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed to allocate the control-plane buffers" << endl << endl;
		return EXIT_FAILURE;
	}

	// ============================================================================
	// Step 5: Set Kernel Arguments and Run the Application
	//         mover j args 0..7 : w_hbm_0 .. w_hbm_7  = HBM[8j .. 8j+7]
	//         ctrl    args 0..2 : x_q, r_g, r_u
	//         collect args 0..1 : c_down, y_q
	// ============================================================================
	// events: one weight upload + (input, output) per case; MLP_KPC kernels per case
	const int Nb_Of_Mem_Events = 1 + 2 * NB_CASES;
	const int Nb_Of_Exe_Events = MLP_KPC * NB_CASES;
	cl_event *Mem_op_event = new cl_event[Nb_Of_Mem_Events];
	cl_event *K_exe_event  = new cl_event[Nb_Of_Exe_Events];

	#ifdef ALL_MESSAGES
	cout << endl;
	cout << "HOST-Info: ============================================================= " << endl;
	cout << "HOST-Info: (Step 5) Run Application                                      " << endl;
	cout << "HOST-Info: ============================================================= " << endl;
	cout << "HOST-Info: Setting Kernel arguments (mover " << MLP_NB_ARGS_MOVER
	     << " x" << MLP_NMV << ", ctrl " << MLP_NB_ARGS_CTRL
	     << ", collect " << MLP_NB_ARGS_COLLECT << ") ..." << endl;
	#endif

	errCode = CL_SUCCESS;

	// Mover CU j gets the MLP_NPC_MV channel images its sp= lines bind it to.
	// The kernel numbers its ports 0..MLP_NPC_MV-1 whatever CU it is, so the
	// host has to apply the same j * MLP_NPC_MV offset the link does -- a
	// mismatch here would feed an engine another block's weights and fail the
	// bit-exact check rather than crash.
	for (int j = 0; j < MLP_NMV; ++j)
		for (int p = 0; p < MLP_NPC_MV; ++p)
			errCode |= clSetKernelArg(K_mover[j], p, sizeof(cl_mem),
			                          &GlobMem_W[j * MLP_NPC_MV + p]);

	// NOTE: AXI4-Stream ports consume kernel argument indices too and must be
	// left unset ("Invalid stream_argument value for kernel arg" otherwise), so
	// every kernel declares its pointers BEFORE its stream ports and only the
	// indices below exist for the host.
	errCode |= clSetKernelArg(K_ctrl, 0, sizeof(cl_mem), &GlobMem_x);
	errCode |= clSetKernelArg(K_ctrl, 1, sizeof(cl_mem), &GlobMem_rg);
	errCode |= clSetKernelArg(K_ctrl, 2, sizeof(cl_mem), &GlobMem_ru);

	errCode |= clSetKernelArg(K_collect, 0, sizeof(cl_mem), &GlobMem_cd);
	errCode |= clSetKernelArg(K_collect, 1, sizeof(cl_mem), &GlobMem_RES);

	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-ERROR: Failed to set Kernel arguments" << endl << endl;
		return EXIT_FAILURE;
	}

	// ------------------------------------------------------------------
	// Step 5.2: Run the golden model for every case FIRST
	//   Doing it here (rather than between kernel launches) keeps the
	//   device timeline free of host compute, so the profiling window in
	//   Step 7 contains device activity only.
	// ------------------------------------------------------------------
	cout << "HOST-Info: Running the golden model for " << NB_CASES
	     << " case(s) ..." << endl;
	{
		static int       g_h [MLP_F];
		static long long g_ps[MLP_N_DOWN];
		for (int c = 0; c < NB_CASES; ++c) {
			mlp_gen_x(&layer, x_q, c);
			mlp_golden(&layer, x_q, r_g, r_u, c_down,
			           g_h, g_ps, gold_y[c], &cov[c]);
			for (int j = 0; j < MLP_N_DOWN; ++j)
				ref_y[c][j] = (double)g_ps[j] * layer.ws_down[j] * MLP_S_H;
			cout << "HOST-Info:   case " << setw(3) << c << "  "
			     << left << setw(10) << mlp_case_name(c) << right
			     << "  y_q [" << cov[c].y_min << ", " << cov[c].y_max << "]"
			     << "  int16 sat " << cov[c].ysat_pos + cov[c].ysat_neg
			     << "  clamps " << cov[c].qg_sat << "/" << cov[c].qu_sat
			     << "/" << cov[c].h_sat
			     << "  LUT " << cov[c].lut_hits << "/256"
			     << "  fq_h_mis " << cov[c].fq_h_mis << endl;
		}
	}

	// ------------------------------------------------------------------
	// Step 5.3: Upload the weights ONCE
	//   14.16 MB of INT2 weights are resident in HBM for the whole run;
	//   only the activation vector is refreshed per case (r_g / r_u / c_down
	//   belong to the layer and are uploaded with it).  This is
	//   also what makes the per-case kernel time a meaningful per-token
	//   number instead of one dominated by a weight reload.
	// ------------------------------------------------------------------
	#ifdef ALL_MESSAGES
	cout << "HOST_Info: Copy weights to Global Memory (once) ..." << endl;
	#endif

	cl_mem W_Buffers[MLP_NPC];
	for (int p = 0; p < MLP_NPC; ++p) W_Buffers[p] = GlobMem_W[p];

	errCode = clEnqueueMigrateMemObjects(Command_Queue, MLP_NPC, W_Buffers, 0, 0, NULL, &Mem_op_event[0]);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed to write the weight buffers to Global Memory" << endl << endl;
		return EXIT_FAILURE;
	}
	clFinish(Command_Queue);

	// ------------------------------------------------------------------
	// Step 5.4: One kernel launch per test case
	// ------------------------------------------------------------------
	cl_mem In_Buffers[4];
	In_Buffers[0] = GlobMem_x;
	In_Buffers[1] = GlobMem_rg;
	In_Buffers[2] = GlobMem_ru;
	In_Buffers[3] = GlobMem_cd;

	for (int c = 0; c < NB_CASES; ++c) {
		// refresh the activation vector in place: the cl_mem objects are
		// CL_MEM_USE_HOST_PTR views of exactly these arrays (the requant
		// vectors are re-migrated unchanged, they are tiny)
		mlp_gen_x(&layer, x_q, c);
		memset(RES, 0, MLP_N_DOWN * sizeof(int));

		errCode = clEnqueueMigrateMemObjects(Command_Queue, 4, In_Buffers, 0, 0, NULL,
		                                     &Mem_op_event[1 + 2 * c]);
		if (errCode != CL_SUCCESS) {
			cout << endl << "Host-Error: Failed to write the input vectors (case " << c << ")" << endl << endl;
			return EXIT_FAILURE;
		}

		errCode = clEnqueueBarrierWithWaitList(Command_Queue, 0, NULL, NULL);
		if (errCode != CL_SUCCESS) {
			cout << endl << "HOST-Error: Failed to Submit BarrierWithWaitList" << endl << endl;
			return EXIT_FAILURE;
		}

		cout << "HOST-Info: Submitting Kernels mlp_collect + mlp_mover x" << MLP_NMV
		     << " + mlp_ctrl, case " << c
		     << " (" << mlp_case_name(c) << ") ..." << endl;

		// The collector is enqueued FIRST and simply blocks on its three
		// partial-sum streams; the movers and the control kernel then start
		// the token.  The queue is out of order, so all six run concurrently
		// -- which they must, the collector drains the engines while the
		// movers are still feeding them.
		errCode = clEnqueueTask(Command_Queue, K_collect, 0, NULL,
		                        &K_exe_event[MLP_KEV_COLLECT(c)]);
		if (errCode != CL_SUCCESS) {
			cout << endl << "HOST-Error: Failed to submit K_collect (case " << c << ")" << endl << endl;
			return EXIT_FAILURE;
		}

		// The movers go in before the control kernel on purpose.  An engine
		// reads its whole control block BEFORE it touches a weight lane, so
		// the movers simply fill their 512-deep lane FIFOs and back off until
		// mlp_ctrl has run; starting them early just hides their launch
		// latency.  There is no deadlock either way -- the control lanes come
		// from a different CU, so a full weight FIFO cannot block them.
		for (int j = 0; j < MLP_NMV; ++j) {
			errCode = clEnqueueTask(Command_Queue, K_mover[j], 0, NULL,
			                        &K_exe_event[MLP_KEV_MOVER(c, j)]);
			if (errCode != CL_SUCCESS) {
				cout << endl << "HOST-Error: Failed to submit K_mover[" << j
				     << "] (case " << c << ")" << endl << endl;
				return EXIT_FAILURE;
			}
		}

		errCode = clEnqueueTask(Command_Queue, K_ctrl, 0, NULL,
		                        &K_exe_event[MLP_KEV_CTRL(c)]);
		if (errCode != CL_SUCCESS) {
			cout << endl << "HOST-Error: Failed to submit K_ctrl (case " << c << ")" << endl << endl;
			return EXIT_FAILURE;
		}

		// y_q is written by the collector alone, so that is the only event the
		// read-back has to wait on.  The clFinish below still fences the
		// movers before the next case rewrites the input buffers.
		errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_RES, CL_MIGRATE_MEM_OBJECT_HOST,
		                                     1, &K_exe_event[MLP_KEV_COLLECT(c)],
		                                     &Mem_op_event[2 + 2 * c]);
		if (errCode != CL_SUCCESS) {
			cout << endl << "Host-Error: Failed to submit Copy Results (case " << c << ")" << endl << endl;
			return EXIT_FAILURE;
		}

		clFinish(Command_Queue);
		memcpy(hw_y[c], RES, MLP_N_DOWN * sizeof(int));
	}

	cout << endl << "HOST_Info: All " << NB_CASES << " case(s) completed" << endl;

	// ============================================================================
	// Step 6: Processing Output Results
	// ============================================================================
	#ifdef ALL_MESSAGES
	cout << endl;
	cout << "HOST-Info: ============================================================= " << endl;
	cout << "HOST-Info: (Step 6) Store and Check the Output Results                   " << endl;
	cout << "HOST-Info: ============================================================= " << endl;
	#endif

	// ------------------------------------------------------
	// Step 6.1: Store every case's output results in RES.txt
	// ------------------------------------------------------
	char Output_File_Name[] = "RES.txt";
	cout << "HOST_Info: Store output results in: " << Output_File_Name << endl;

	fstream RES_File;
	RES_File.open(Output_File_Name, ios::out);
	if (! RES_File.is_open()) {
		cout << endl << "HOST-Error: Failed to open the " << Output_File_Name << " file for write" << endl << endl;
		return EXIT_FAILURE;
	}

	for (int c = 0; c < NB_CASES; c++) {
		RES_File << "# case " << c << "  " << mlp_case_name(c)
		         << "  S_Y16=" << scientific << setprecision(6) << MLP_S_Y16
		         << fixed << endl;
		RES_File << "#     j   hw_code gold_code   hw_code*S_Y16   p*ws_down*S_H" << endl;
		for (int j = 0; j < MLP_N_DOWN; j++) {
			RES_File << setw(7) << j
			         << setw(10) << hw_y[c][j]
			         << setw(10) << gold_y[c][j]
			         << setw(16) << fixed << setprecision(6) << hw_y[c][j] * MLP_S_Y16
			         << setw(16) << ref_y[c][j]
			         << endl;
		}
		RES_File << endl;
	}
	RES_File.close();

	// ------------------------------------------------------
	// Step 6.2: Check correctness, case by case
	// ------------------------------------------------------
	bool   error_detected = false;
	int    Max_Number_Of_Failures = 5;
	int    nb_failed_cases = 0;

	cout << endl << "Host-Info: =============================================================" << endl;
	cout <<         "Host-Info: Verifying the down_proj INT16 codes vs. the golden model"      << endl;
	cout <<         "Host-Info:   integer datapath end to end -> bit-exact match required"     << endl;
	cout <<         "Host-Info:   semantic: |y_q - p*ws_down*S_H/S_Y16| <= 0.5 + |p|*2^-25"   << endl;
	cout <<         "Host-Info: " << MLP_N_DOWN << " outputs per case (INT16, 1 LSB = S_Y16 = "
	     << scientific << setprecision(3) << MLP_S_Y16 << ")" << fixed << endl;
	cout << "Host-Info: ------------------------------------------------------------------------------" << endl;
	cout << "Host-Info: case  name        mismatch  max|diff|   y_min   y_max   sat(+/-)  sem_err  res" << endl;
	cout << "Host-Info:                              (codes)                               (LSB)" << endl;
	cout << "Host-Info: ------------------------------------------------------------------------------" << endl;

	for (int c = 0; c < NB_CASES; c++) {
		int nb_mismatch = 0, max_diff = 0, nb_printed = 0;

		for (int j = 0; j < MLP_N_DOWN; j++) {
			const int diff = abs(hw_y[c][j] - gold_y[c][j]);
			if (diff > max_diff) max_diff = diff;
			if (diff != 0) {
				++nb_mismatch;
				if (nb_printed++ < Max_Number_Of_Failures)
					cout << "Host-Info:   case " << c << " j=" << setw(5) << j
					     << "  expected " << setw(6) << gold_y[c][j]
					     << "  actual " << setw(6) << hw_y[c][j] << "   Error" << endl;
			}
		}

		const bool sem_ok   = cov[c].sem_err <= cov[c].sem_bound;
		const bool fq_ok    = cov[c].fq_h_mis <= MLP_F / 100 && cov[c].fq_h_maxd <= 8;
		const bool case_bad = (nb_mismatch != 0) || !sem_ok || !fq_ok;

		cout << "Host-Info: " << setw(4) << c << "  " << left << setw(10)
		     << mlp_case_name(c) << right
		     << setw(10) << nb_mismatch
		     << setw(11) << max_diff
		     << setw(8)  << cov[c].y_min
		     << setw(8)  << cov[c].y_max
		     << setw(6)  << cov[c].ysat_pos << "/" << left << setw(5) << cov[c].ysat_neg << right
		     << setw(8)  << fixed << setprecision(3) << cov[c].sem_err
		     << "  " << (case_bad ? "FAIL" : "pass") << endl;

		if (case_bad) { error_detected = true; ++nb_failed_cases; }
	}

	cout << "Host-Info: ------------------------------------------------------------------------------" << endl;
	cout << "Host-Info: =============================================================" << endl;

	if (error_detected == false) {
		cout << "Host-Info: Test Successful (" << NB_CASES << "/" << NB_CASES << " cases)" << endl;
	} else {
		cout << "Host-Error: Test Failed (" << nb_failed_cases << "/" << NB_CASES
		     << " cases over tolerance)" << endl;
	}

	// ============================================================================
	// Step 7: Custom Profiling
	// ============================================================================
	cout << "HOST-Info: ============================================================= " << endl;
	cout << "HOST-Info: (Step 7) Custom Profiling                                     " << endl;
	cout << "HOST-Info: ============================================================= " << endl;
	int Nb_Of_Kernels         = Nb_Of_Exe_Events;
	int Nb_Of_Memory_Tranfers = Nb_Of_Mem_Events;

	// Transfer_1        : the 14.16 MB weight upload, done once
	// Transfer_2,3      : case 0 input vectors / output readback
	// Transfer_4,5      : case 1 ... and so on
	cout << "HOST-Info: Transfer_1 is the one-off " << MLP_NPC * (long)MLP_CH_BYTES / (1024 * 1024)
	     << " MB weight upload; Transfer_2k/2k+1 are case k-1 in/out." << endl;
	cout << "HOST-Info: Per-token latency is the SLOWEST K_mv*_c[c] (they run"  << endl;
	cout << "HOST-Info: concurrently and self-align through the lane FIFOs; the" << endl;
	cout << "HOST-Info: collector and mlp_ctrl overlap them)" << endl;
	cout << "HOST-Info: -- the weight upload is amortized over every token and must" << endl;
	cout << "HOST-Info: NOT be counted per case." << endl;

	string *list_of_kernel_names = new string[Nb_Of_Exe_Events];
	for (int c = 0; c < NB_CASES; ++c) {
		list_of_kernel_names[MLP_KEV_COLLECT(c)] = "K_collect_c" + to_string(c);
		list_of_kernel_names[MLP_KEV_CTRL(c)]    = "K_ctrl_c"    + to_string(c);
		for (int j = 0; j < MLP_NMV; ++j)
			list_of_kernel_names[MLP_KEV_MOVER(c, j)] =
				"K_mv" + to_string(j) + "_c" + to_string(c);
	}
	run_custom_profiling(Nb_Of_Kernels, Nb_Of_Memory_Tranfers, K_exe_event, Mem_op_event, list_of_kernel_names);
	delete[] list_of_kernel_names;

	// ============================================================================
	// Step 8: Release Allocated Resources
	// ============================================================================
	clReleaseDevice(Target_Device_ID);

	for (int i = 0; i < Nb_Of_Mem_Events; i++) clReleaseEvent(Mem_op_event[i]);
	for (int i = 0; i < Nb_Of_Exe_Events; i++) clReleaseEvent(K_exe_event[i]);

	for (int p = 0; p < MLP_NPC; ++p) clReleaseMemObject(GlobMem_W[p]);
	clReleaseMemObject(GlobMem_x);
	clReleaseMemObject(GlobMem_rg);
	clReleaseMemObject(GlobMem_ru);
	clReleaseMemObject(GlobMem_cd);
	clReleaseMemObject(GlobMem_RES);

	clReleaseKernel(K_ctrl);
	for (int j = 0; j < MLP_NMV; ++j) clReleaseKernel(K_mover[j]);
	clReleaseKernel(K_collect);
	clReleaseProgram(Program);
	clReleaseCommandQueue(Command_Queue);
	clReleaseContext(Context);

	delete[] Platform_IDs;
	delete[] Device_IDs;
	delete[] Mem_op_event;
	delete[] K_exe_event;
	for (int c = 0; c < NB_CASES; ++c) { delete[] gold_y[c]; delete[] hw_y[c]; delete[] ref_y[c]; }
	delete[] gold_y;  delete[] hw_y;  delete[] ref_y;  delete[] cov;
	mlp_free_layer(&layer);
	for (int p = 0; p < MLP_NPC; ++p) free(ch_img[p]);
	free(x_q); free(r_g); free(r_u);
	free(c_down); free(RES);

	cout << endl << "HOST-Info: DONE" << endl << endl;

	return (error_detected ? EXIT_FAILURE : EXIT_SUCCESS);
}
