/*******************************************************************************
** HOST code for the Gemma-4-E2B MLP accelerator (Alveo U280, Vitis / OpenCL)
**
**   o) detects the Xilinx platform / target device, loads the xclbin
**   o) repacks the INT2 QAT weights into the 32 HBM pseudo-channel images
**      (mlp_model.h -- the same offline layout the C testbench uses):
**        - W_gate / W_up : fused column zip, ^0xAA, 512-column macro tiles,
**                          SLR(t) = t/16, 16 columns per pseudo-channel
**        - W_down        : K split, SLR(n_tile,k_tile) = k_tile/8,
**                          4 rows per pseudo-channel
**   o) pins every channel image to its own HBM bank (XCL_MEM_TOPOLOGY), which
**      is what lets the SLR0 data mover own all 32 AXI ports locally
**   o) runs the "mlp" kernel and checks the 1536-wide Q7.8 output against the
**      golden model (exact integer GEMV / requant / GELU-LUT path)
**
** The v++ link must place the ports on matching banks, e.g.
**     [connectivity]
**     sp=mlp_1.w_hbm_0:HBM[0]
**     ...
**     sp=mlp_1.w_hbm_31:HBM[31]
**     sp=mlp_1.x_q:HBM[0]      (and the rest of the gmemS bundle)
**     slr=mlp_1:SLR0
**
** The real QAT export is not available yet, so all weights and scales are
** chosen placeholder values and only functional correctness is checked.
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
// Pass/fail thresholds on the Q7.8 output.  The bit-accurate C simulation
// measured MAE = 0.257 LSB and max |err| = 0.501 LSB (the output rounding
// floor); the thresholds below keep a 2x / 4x margin.
// ---------------------------------------------------------------------------
static const double LSB         = 1.0 / MLP_HID_SCALE;
static const double MAX_ABS_TOL = 4.0 * LSB;
static const double MAE_TOL     = 1.0 * LSB;

// number of kernel arguments: 32 HBM channels + 7 control-plane vectors
#define MLP_NB_ARGS (MLP_NPC + 7)

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

	if (argc != 4 && argc != 5)
	{
		cout << "HOST-Error: Incorrect command line syntax " << endl;
		cout << "HOST-Info:  Usage: " << argv[0] << " <Platform_Vendor> <Device_Name> <XCLBIN_File> [Nb_Of_Test_Cases]" << endl << endl;
		return EXIT_FAILURE;
	}

	const char* Target_Platform_Vendor   = argv[1];
	const char* Target_Device_Name       = argv[2];
	const char* xclbinFilename           = argv[3];

	// One test case = one draw of the six runtime vectors (mlp_model.h,
	// MLP_TC_*).  The weights are NOT part of a case: they stay resident in
	// HBM, exactly as at decode time where only the activation changes per
	// token.  sw_emu is slow, so the count is settable from the command line.
	const int NB_CASES = (argc == 5) ? atoi(argv[4]) : MLP_NB_TESTS;
	if (NB_CASES < 1) {
		cout << "HOST-Error: Nb_Of_Test_Cases must be >= 1" << endl << endl;
		return EXIT_FAILURE;
	}

	cout << "HOST-Info: Platform_Vendor   : " << Target_Platform_Vendor << endl;
	cout << "HOST-Info: Device_Name       : " << Target_Device_Name << endl;
	cout << "HOST-Info: XCLBIN_file       : " << xclbinFilename << endl;
	cout << "HOST-Info: Model             : Gemma-4-E2B MLP, K=" << MLP_K
	     << " F=" << MLP_F << ", INT2 weights, decode stage" << endl;
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

	cl_kernel K_mlp;

	#ifdef ALL_MESSAGES
	cout << "HOST-Info: Creating a Kernel: mlp ..." << endl;
	#endif
	K_mlp = clCreateKernel(Program, "mlp", &errCode);
	if (errCode != CL_SUCCESS) {
		cout << endl << "HOST-Error: Failed to create K_mlp" << endl << endl;
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

	int *x_q      = (int *)aligned_alloc_or_die(MLP_K        * sizeof(int), "x_q");
	int *r_g      = (int *)aligned_alloc_or_die(MLP_F        * sizeof(int), "r_g");
	int *r_u      = (int *)aligned_alloc_or_die(MLP_F        * sizeof(int), "r_u");
	int *c_down   = (int *)aligned_alloc_or_die(MLP_N_DOWN   * sizeof(int), "c_down");
	int *ln_gamma = (int *)aligned_alloc_or_die(MLP_N_DOWN   * sizeof(int), "ln_gamma");
	int *resid    = (int *)aligned_alloc_or_die(MLP_N_DOWN   * sizeof(int), "resid");
	int *RES      = (int *)aligned_alloc_or_die(MLP_N_DOWN   * sizeof(int), "RES");
	memset(RES, 0, MLP_N_DOWN * sizeof(int));

	// per-case golden outputs, hardware outputs and coverage counters.  The
	// golden model is run for EVERY case up front so that no host-side compute
	// sits between two kernel launches and pollutes the profiling window.
	double    **gold_y = new double *[NB_CASES];
	int       **hw_y   = new int    *[NB_CASES];
	mlp_cov_t  *cov    = new mlp_cov_t[NB_CASES];
	for (int c = 0; c < NB_CASES; ++c) {
		gold_y[c] = new double[MLP_N_DOWN];
		hw_y[c]   = new int[MLP_N_DOWN];
	}

	cout << "HOST-Info: Generating placeholder weights / scales ... ";
	mlp_gen_stimulus(x_q, r_g, r_u, c_down, ln_gamma, resid, MLP_TC_BASELINE);
	cout << "done" << endl;

	// mlp_repack_* fill one contiguous [32][MLP_CH_BYTES] block, so the images
	// are repacked into a staging buffer and then split across the 32
	// page-aligned per-channel allocations that back the HBM buffers.
	cout << "HOST-Info: Repacking INT2 weights into " << MLP_NPC
	     << " HBM channel images (" << MLP_NPC * (long)MLP_CH_BYTES / (1024 * 1024)
	     << " MB) ... ";
	cout.flush();
	{
		static mlp_img_t staging;
		mlp_repack_gate_up(staging);
		mlp_repack_down(staging);
		cout << "done" << endl;
		if (mlp_check_packing(staging)) {
			cout << "HOST-Error: HBM packing contract self-check FAILED" << endl;
			return EXIT_FAILURE;
		}
		cout << "HOST-Info: packing contract self-check  : ok" << endl;
		for (int p = 0; p < MLP_NPC; ++p)
			memcpy(ch_img[p], staging[p], MLP_CH_BYTES);
	}

	// ------------------------------------------------------------------
	// Step 4.2: Create Buffers in Global Memory
	//   every weight image is pinned to its own HBM pseudo-channel so that
	//   the SLR0 data mover reaches all 32 AXI ports locally (zero SLL)
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
	cl_mem GlobMem_ln  = clCreateBuffer(Context, CL_MEM_READ_ONLY  | CL_MEM_USE_HOST_PTR, MLP_N_DOWN * sizeof(int), ln_gamma, &errCode);
	cl_mem GlobMem_rs  = clCreateBuffer(Context, CL_MEM_READ_ONLY  | CL_MEM_USE_HOST_PTR, MLP_N_DOWN * sizeof(int), resid,    &errCode);
	cl_mem GlobMem_RES = clCreateBuffer(Context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, MLP_N_DOWN * sizeof(int), RES,      &errCode);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed to allocate the control-plane buffers" << endl << endl;
		return EXIT_FAILURE;
	}

	// ============================================================================
	// Step 5: Set Kernel Arguments and Run the Application
	//         args 0..31 : w_hbm_0 .. w_hbm_31
	//         args 32..38: x_q, r_g, r_u, c_down, ln_gamma, resid, y_out
	// ============================================================================
	// events: one weight upload + (input, output) per case; one kernel per case
	const int Nb_Of_Mem_Events = 1 + 2 * NB_CASES, Nb_Of_Exe_Events = NB_CASES;
	cl_event *Mem_op_event = new cl_event[Nb_Of_Mem_Events];
	cl_event *K_exe_event  = new cl_event[Nb_Of_Exe_Events];

	#ifdef ALL_MESSAGES
	cout << endl;
	cout << "HOST-Info: ============================================================= " << endl;
	cout << "HOST-Info: (Step 5) Run Application                                      " << endl;
	cout << "HOST-Info: ============================================================= " << endl;
	cout << "HOST-Info: Setting Kernel arguments (" << MLP_NB_ARGS << ") ..." << endl;
	#endif

	errCode = CL_SUCCESS;
	for (int p = 0; p < MLP_NPC; ++p)
		errCode |= clSetKernelArg(K_mlp, p, sizeof(cl_mem), &GlobMem_W[p]);
	errCode |= clSetKernelArg(K_mlp, MLP_NPC + 0, sizeof(cl_mem), &GlobMem_x);
	errCode |= clSetKernelArg(K_mlp, MLP_NPC + 1, sizeof(cl_mem), &GlobMem_rg);
	errCode |= clSetKernelArg(K_mlp, MLP_NPC + 2, sizeof(cl_mem), &GlobMem_ru);
	errCode |= clSetKernelArg(K_mlp, MLP_NPC + 3, sizeof(cl_mem), &GlobMem_cd);
	errCode |= clSetKernelArg(K_mlp, MLP_NPC + 4, sizeof(cl_mem), &GlobMem_ln);
	errCode |= clSetKernelArg(K_mlp, MLP_NPC + 5, sizeof(cl_mem), &GlobMem_rs);
	errCode |= clSetKernelArg(K_mlp, MLP_NPC + 6, sizeof(cl_mem), &GlobMem_RES);

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
			mlp_gen_stimulus(x_q, r_g, r_u, c_down, ln_gamma, resid, c);
			mlp_golden(x_q, r_g, r_u, c_down, ln_gamma, resid,
			           g_h, g_ps, gold_y[c], &cov[c]);
			cout << "HOST-Info:   case " << setw(3) << c << "  "
			     << left << setw(10) << mlp_case_name(c) << right
			     << "  mean-sq " << scientific << setprecision(3) << cov[c].ms
			     << "  rsqrt " << cov[c].rs
			     << "  clamps " << fixed << cov[c].qg_sat << "/" << cov[c].qu_sat
			     << "/" << cov[c].h_sat
			     << "  LUT " << cov[c].lut_hits << "/256" << endl;
		}
	}

	// ------------------------------------------------------------------
	// Step 5.3: Upload the weights ONCE
	//   14.16 MB of INT2 weights are resident in HBM for the whole run;
	//   only the 6 narrow runtime vectors are refreshed per case.  This is
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
	cl_mem In_Buffers[6];
	In_Buffers[0] = GlobMem_x;
	In_Buffers[1] = GlobMem_rg;
	In_Buffers[2] = GlobMem_ru;
	In_Buffers[3] = GlobMem_cd;
	In_Buffers[4] = GlobMem_ln;
	In_Buffers[5] = GlobMem_rs;

	for (int c = 0; c < NB_CASES; ++c) {
		// refresh the activation / scale vectors in place: the cl_mem
		// objects are CL_MEM_USE_HOST_PTR views of exactly these arrays
		mlp_gen_stimulus(x_q, r_g, r_u, c_down, ln_gamma, resid, c);
		memset(RES, 0, MLP_N_DOWN * sizeof(int));

		errCode = clEnqueueMigrateMemObjects(Command_Queue, 6, In_Buffers, 0, 0, NULL,
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

		cout << "HOST-Info: Submitting Kernel mlp, case " << c
		     << " (" << mlp_case_name(c) << ") ..." << endl;

		errCode = clEnqueueTask(Command_Queue, K_mlp, 0, NULL, &K_exe_event[c]);
		if (errCode != CL_SUCCESS) {
			cout << endl << "HOST-Error: Failed to submit K_mlp (case " << c << ")" << endl << endl;
			return EXIT_FAILURE;
		}

		errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_RES, CL_MIGRATE_MEM_OBJECT_HOST,
		                                     1, &K_exe_event[c], &Mem_op_event[2 + 2 * c]);
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
		         << "  mean_sq=" << scientific << setprecision(6) << cov[c].ms
		         << "  rsqrt=" << cov[c].rs << fixed << endl;
		RES_File << "#     j        hw_out         golden          error" << endl;
		for (int j = 0; j < MLP_N_DOWN; j++) {
			double hw   = (double)hw_y[c][j] / (double)MLP_HID_SCALE;
			double gold = mlp_sat_q78(gold_y[c][j]);
			RES_File << setw(7) << j
			         << setw(15) << fixed << setprecision(6) << hw
			         << setw(15) << gold
			         << setw(15) << scientific << setprecision(3) << (hw - gold)
			         << fixed << endl;
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
	double worst_mae = 0.0, worst_max = 0.0;
	int    worst_mae_c = 0, worst_max_c = 0;

	cout << endl << "Host-Info: =============================================================" << endl;
	cout <<         "Host-Info: Verifying the MLP output vs. the golden model"                 << endl;
	cout <<         "Host-Info:   exact integer GEMV / requant / GELU-LUT path,"               << endl;
	cout <<         "Host-Info:   double-precision dequant + post_feedforward_layernorm"       << endl;
	cout <<         "Host-Info:   tolerance: max|err| <= " << scientific << setprecision(3) << MAX_ABS_TOL
	     <<         ", MAE <= " << MAE_TOL << fixed << endl;
	cout <<         "Host-Info: " << MLP_N_DOWN << " outputs per case (Q7.8, 1 LSB = "
	     << scientific << setprecision(3) << LSB << ")" << fixed << endl;
	cout << "Host-Info: ---------------------------------------------------------------------------" << endl;
	cout << "Host-Info: case  name          MAE      RMSE       max   worst_j   mean-sq    rsqrt  res" << endl;
	cout << "Host-Info:                    (LSB)     (LSB)     (LSB)" << endl;
	cout << "Host-Info: ---------------------------------------------------------------------------" << endl;

	for (int c = 0; c < NB_CASES; c++) {
		double max_abs_err = 0.0, sum_abs_err = 0.0, sum_sq_err = 0.0;
		int    worst_j = 0, nb_printed = 0;
		bool   case_bad = false;

		for (int j = 0; j < MLP_N_DOWN; j++) {
			double hw   = (double)hw_y[c][j] / (double)MLP_HID_SCALE;
			double gold = mlp_sat_q78(gold_y[c][j]);
			double err  = fabs(hw - gold);

			sum_abs_err += err;
			sum_sq_err  += err * err;
			if (err > max_abs_err) { max_abs_err = err; worst_j = j; }

			if (err > MAX_ABS_TOL) {
				case_bad = true;
				if (nb_printed++ < Max_Number_Of_Failures)
					cout << "Host-Info:   case " << c << " j=" << setw(5) << j
					     << "  expected " << setw(12) << setprecision(6) << gold
					     << "  actual " << setw(12) << hw << "   Error" << endl;
			}
		}

		double mae  = sum_abs_err / MLP_N_DOWN;
		double rmse = sqrt(sum_sq_err / MLP_N_DOWN);
		if (mae > MAE_TOL) case_bad = true;

		cout << "Host-Info: " << setw(4) << c << "  " << left << setw(11)
		     << mlp_case_name(c) << right
		     << setw(8) << fixed << setprecision(3) << mae / LSB
		     << setw(10) << rmse / LSB
		     << setw(10) << max_abs_err / LSB
		     << setw(9)  << worst_j
		     << setw(11) << scientific << setprecision(2) << cov[c].ms
		     << setw(10) << cov[c].rs << fixed
		     << "  " << (case_bad ? "FAIL" : "pass") << endl;

		if (case_bad) { error_detected = true; ++nb_failed_cases; }
		if (mae / LSB > worst_mae)         { worst_mae = mae / LSB; worst_mae_c = c; }
		if (max_abs_err / LSB > worst_max) { worst_max = max_abs_err / LSB; worst_max_c = c; }
	}

	cout << "Host-Info: ---------------------------------------------------------------------------" << endl;
	cout << "Host-Info: worst MAE       : " << fixed << setprecision(3) << worst_mae
	     << " LSB  (case " << worst_mae_c << " " << mlp_case_name(worst_mae_c) << ")" << endl;
	cout << "Host-Info: worst max |err| : " << worst_max
	     << " LSB  (case " << worst_max_c << " " << mlp_case_name(worst_max_c) << ")" << endl;
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
	cout << "HOST-Info: Per-token latency is K_mlp[c] alone -- the weight upload is"  << endl;
	cout << "HOST-Info: amortized over every token and must NOT be counted per case." << endl;

	string *list_of_kernel_names = new string[Nb_Of_Exe_Events];
	for (int c = 0; c < NB_CASES; ++c)
		list_of_kernel_names[c] = "K_mlp_c" + to_string(c);
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
	clReleaseMemObject(GlobMem_ln);
	clReleaseMemObject(GlobMem_rs);
	clReleaseMemObject(GlobMem_RES);

	clReleaseKernel(K_mlp);
	clReleaseProgram(Program);
	clReleaseCommandQueue(Command_Queue);
	clReleaseContext(Context);

	delete[] Platform_IDs;
	delete[] Device_IDs;
	delete[] Mem_op_event;
	delete[] K_exe_event;
	for (int c = 0; c < NB_CASES; ++c) { delete[] gold_y[c]; delete[] hw_y[c]; }
	delete[] gold_y;  delete[] hw_y;  delete[] cov;
	for (int p = 0; p < MLP_NPC; ++p) free(ch_img[p]);
	free(x_q); free(r_g); free(r_u);
	free(c_down); free(ln_gamma); free(resid); free(RES);

	cout << endl << "HOST-Info: DONE" << endl << endl;

	return (error_detected ? EXIT_FAILURE : EXIT_SUCCESS);
}
