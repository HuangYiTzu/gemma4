/*******************************************************************************
** HOST code for standalone RMSNorm hardware test (Alveo U280, Vitis / OpenCL)
**
**   o) detects the Xilinx platform / target device, loads the xclbin
**   o) runs pre_feedforward_layernorm_hls / post_feedforward_layernorm_hls on
**      every test vector already baked into tb_rmsnorm_vectors.h (same file
**      the C-simulation testbench tb_rmsnorm.cpp uses -- full 35 layers x 3
**      tokens of real QAT activations, with bit-accurate expected outputs)
**   o) compares the hardware output against TB_PRE_EXPECTED / TB_POST_EXPECTED
**      bit-exactly (same pass/fail criterion as tb_rmsnorm.cpp: any nonzero
**      diff is a fail -- this is checking C++/RTL correctness, not
**      quantization accuracy, which was already validated separately)
**
** This is deliberately NOT modeled on MLP's host.cpp beyond the basic OpenCL
** platform/device/program boilerplate: RMSNorm has no HBM weight loading, no
** multi-CU movers, just two simple kernels with one input buffer, one scalar
** layer_idx, and one output buffer each. Each test case is one clEnqueueTask
** per kernel.
**
** Build against the xclbin produced by a STANDALONE RMSNorm Vitis project
** (its own kernel + hw_link + host project, per the plan: test RMSNorm's own
** correctness on real hardware before resolving the MLP interface question).
*******************************************************************************/

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <iomanip>

using namespace std;

#include <CL/cl.h>
#include <CL/cl_ext.h>

// Reuses the exact same test vectors tb_rmsnorm.cpp (C-simulation) already
// validated against: TB_NUM_PRE_CASES/TB_NUM_POST_CASES, TB_HIDDEN_SIZE,
// TB_PRE_LAYER_IDX/TB_POST_LAYER_IDX, TB_PRE_INPUT/TB_POST_INPUT,
// TB_PRE_EXPECTED/TB_POST_EXPECTED.
#include "tb_rmsnorm_vectors.h"

// ---------------------------------------------------------------------------
// Minimal loadFile2Memory (same contract as MLP's help_functions.cpp version,
// reimplemented here so this host has no dependency on MLP's helper files)
// ---------------------------------------------------------------------------
static int loadFile2Memory(const char *filename, char **result)
{
	FILE *f = fopen(filename, "rb");
	if (!f) { *result = NULL; return -1; }
	fseek(f, 0, SEEK_END);
	long size = ftell(f);
	fseek(f, 0, SEEK_SET);
	*result = (char *)malloc(size);
	if (fread(*result, 1, size, f) != (size_t)size) { fclose(f); free(*result); *result = NULL; return -1; }
	fclose(f);
	return (int)size;
}

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

// ***************************************************************************
// M A I N
// ***************************************************************************
int main(int argc, char *argv[])
{
	cout << endl;

	// ===========================================================================
	// Step 1: Command line
	// ===========================================================================
	if (argc < 4 || argc > 5) {
		cout << "HOST-Error: Incorrect command line syntax" << endl;
		cout << "HOST-Info:  Usage: " << argv[0]
		     << " <Platform_Vendor> <Device_Name> <XCLBIN_File> [Nb_Of_Cases_Cap]" << endl << endl;
		return EXIT_FAILURE;
	}
	const char *Target_Platform_Vendor = argv[1];
	const char *Target_Device_Name     = argv[2];
	const char *xclbinFilename         = argv[3];
	// RMSNorm has no FPGA DMA weight load to amortize, so testing a handful
	// of cases is plenty for emulation; hardware can run all of them.
	const int NB_PRE_CASES  = (argc >= 5) ? std::min(TB_NUM_PRE_CASES,  atoi(argv[4])) : TB_NUM_PRE_CASES;
	const int NB_POST_CASES = (argc >= 5) ? std::min(TB_NUM_POST_CASES, atoi(argv[4])) : TB_NUM_POST_CASES;

	cout << "HOST-Info: Platform_Vendor : " << Target_Platform_Vendor << endl;
	cout << "HOST-Info: Device_Name     : " << Target_Device_Name << endl;
	cout << "HOST-Info: XCLBIN_file     : " << xclbinFilename << endl;
	cout << "HOST-Info: PRE test cases  : " << NB_PRE_CASES  << " / " << TB_NUM_PRE_CASES  << endl;
	cout << "HOST-Info: POST test cases : " << NB_POST_CASES << " / " << TB_NUM_POST_CASES << endl;

	// ===========================================================================
	// Step 2: Platform / device / context / command queue
	// ===========================================================================
	cl_int errCode;
	cl_uint Nb_Of_Platforms;
	errCode = clGetPlatformIDs(0, NULL, &Nb_Of_Platforms);
	if (errCode != CL_SUCCESS || Nb_Of_Platforms == 0) {
		cout << endl << "HOST-Error: Failed to get the number of available platforms" << endl;
		return EXIT_FAILURE;
	}
	cl_platform_id *Platform_IDs = new cl_platform_id[Nb_Of_Platforms];
	clGetPlatformIDs(Nb_Of_Platforms, Platform_IDs, NULL);

	cl_platform_id Target_Platform_ID = nullptr;
	for (cl_uint i = 0; i < Nb_Of_Platforms; i++) {
		size_t size;
		clGetPlatformInfo(Platform_IDs[i], CL_PLATFORM_VENDOR, 0, NULL, &size);
		char *info = new char[size];
		clGetPlatformInfo(Platform_IDs[i], CL_PLATFORM_VENDOR, size, info, NULL);
		if (strcmp(info, Target_Platform_Vendor) == 0) Target_Platform_ID = Platform_IDs[i];
		delete[] info;
	}
	if (!Target_Platform_ID) {
		cout << endl << "HOST-Error: Failed to detect " << Target_Platform_Vendor << " platform" << endl;
		return EXIT_FAILURE;
	}

	cl_uint Nb_Of_Devices;
	clGetDeviceIDs(Target_Platform_ID, CL_DEVICE_TYPE_ALL, 0, NULL, &Nb_Of_Devices);
	cl_device_id *Device_IDs = new cl_device_id[Nb_Of_Devices];
	clGetDeviceIDs(Target_Platform_ID, CL_DEVICE_TYPE_ALL, Nb_Of_Devices, Device_IDs, NULL);

	cl_device_id Target_Device_ID = nullptr;
	for (cl_uint i = 0; i < Nb_Of_Devices; i++) {
		size_t size;
		clGetDeviceInfo(Device_IDs[i], CL_DEVICE_NAME, 0, NULL, &size);
		char *info = new char[size];
		clGetDeviceInfo(Device_IDs[i], CL_DEVICE_NAME, size, info, NULL);
		if (strcmp(info, Target_Device_Name) == 0) Target_Device_ID = Device_IDs[i];
		delete[] info;
	}
	if (!Target_Device_ID) {
		cout << endl << "HOST-Error: Failed to detect " << Target_Device_Name << " device" << endl;
		return EXIT_FAILURE;
	}

	cl_context Context = clCreateContext(0, 1, &Target_Device_ID, NULL, NULL, &errCode);
	// 2026-10-14修正：原本用CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE，但下面
	// migrate-in -> task -> migrate-out這三個指令之間沒有用event指定依賴，
	// out-of-order queue不保證它們照enqueue順序執行——kernel可能在輸入
	// 資料搬完前就開始跑，或輸出migrate在kernel跑完前就先把舊資料搬回host。
	// 這支host本來就是單純循序跑，不需要靠多CU平行度搶效能，改用預設的
	// in-order queue最簡單：同一個queue裡的指令保證嚴格按enqueue順序執行，
	// 不用額外處理event。
	cl_command_queue Command_Queue = clCreateCommandQueue(
		Context, Target_Device_ID,
		CL_QUEUE_PROFILING_ENABLE, &errCode);

	// ===========================================================================
	// Step 3: Program and kernels
	// ===========================================================================
	char *xclbin_Memory;
	int program_length = loadFile2Memory(xclbinFilename, &xclbin_Memory);
	if (program_length < 0) {
		cout << endl << "HOST-Error: Failed to load " << xclbinFilename << endl;
		return EXIT_FAILURE;
	}
	size_t Program_Length_in_Bytes = program_length;
	cl_program Program = clCreateProgramWithBinary(
		Context, 1, &Target_Device_ID, &Program_Length_in_Bytes,
		(const unsigned char **)&xclbin_Memory, NULL, &errCode);
	if (errCode != CL_SUCCESS) {
		cout << endl << "HOST-Error: Failed to create Program from Binary" << endl;
		return EXIT_FAILURE;
	}
	errCode = clBuildProgram(Program, 1, &Target_Device_ID, NULL, NULL, NULL);
	if (errCode != CL_SUCCESS) {
		cout << endl << "HOST-Error: Failed to build Program" << endl;
		return EXIT_FAILURE;
	}

	cl_kernel K_pre  = clCreateKernel(Program, "pre_feedforward_layernorm_hls",  &errCode);
	if (errCode != CL_SUCCESS) { cout << "HOST-Error: Failed to create K_pre"  << endl; return EXIT_FAILURE; }
	cl_kernel K_post = clCreateKernel(Program, "post_feedforward_layernorm_hls", &errCode);
	if (errCode != CL_SUCCESS) { cout << "HOST-Error: Failed to create K_post" << endl; return EXIT_FAILURE; }

	// ===========================================================================
	// Step 4: Buffers
	//   PRE : residual_in (hid16_t = INT16) -> y_out (act8_t = INT8)
	//   POST: down_proj_mac_out (acc32_t = INT32) -> y_out (hid16_t = INT16)
	// ===========================================================================
	int16_t *pre_in   = (int16_t *)aligned_alloc_or_die(TB_HIDDEN_SIZE * sizeof(int16_t), "pre_in");
	int8_t  *pre_out  = (int8_t  *)aligned_alloc_or_die(TB_HIDDEN_SIZE * sizeof(int8_t),  "pre_out");
	int32_t *post_in  = (int32_t *)aligned_alloc_or_die(TB_HIDDEN_SIZE * sizeof(int32_t), "post_in");
	int16_t *post_out = (int16_t *)aligned_alloc_or_die(TB_HIDDEN_SIZE * sizeof(int16_t), "post_out");

	cl_mem Buf_pre_in   = clCreateBuffer(Context, CL_MEM_READ_ONLY  | CL_MEM_USE_HOST_PTR, TB_HIDDEN_SIZE * sizeof(int16_t), pre_in,   &errCode);
	cl_mem Buf_pre_out  = clCreateBuffer(Context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, TB_HIDDEN_SIZE * sizeof(int8_t),  pre_out,  &errCode);
	cl_mem Buf_post_in  = clCreateBuffer(Context, CL_MEM_READ_ONLY  | CL_MEM_USE_HOST_PTR, TB_HIDDEN_SIZE * sizeof(int32_t), post_in,  &errCode);
	cl_mem Buf_post_out = clCreateBuffer(Context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, TB_HIDDEN_SIZE * sizeof(int16_t), post_out, &errCode);
	if (errCode != CL_SUCCESS) {
		cout << endl << "HOST-Error: Failed to allocate buffers" << endl;
		return EXIT_FAILURE;
	}

	// argument order follows the C++ declaration: (array, layer_idx, array)
	clSetKernelArg(K_pre,  0, sizeof(cl_mem), &Buf_pre_in);
	clSetKernelArg(K_pre,  2, sizeof(cl_mem), &Buf_pre_out);
	clSetKernelArg(K_post, 0, sizeof(cl_mem), &Buf_post_in);
	clSetKernelArg(K_post, 2, sizeof(cl_mem), &Buf_post_out);

	// ===========================================================================
	// Step 5: Run every PRE case, then every POST case
	// ===========================================================================
	cout << endl << "HOST-Info: ======================= Running PRE cases =======================" << endl;
	int pre_fail = 0, pre_max_diff = 0;
	for (int c = 0; c < NB_PRE_CASES; c++) {
		int layer_idx = TB_PRE_LAYER_IDX[c];
		for (int i = 0; i < TB_HIDDEN_SIZE; i++) pre_in[i] = (int16_t)TB_PRE_INPUT[c][i];

		clSetKernelArg(K_pre, 1, sizeof(int), &layer_idx);
		errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &Buf_pre_in, 0, 0, NULL, NULL);
		errCode = clEnqueueTask(Command_Queue, K_pre, 0, NULL, NULL);
		errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &Buf_pre_out, CL_MIGRATE_MEM_OBJECT_HOST, 0, NULL, NULL);
		clFinish(Command_Queue);

		bool case_ok = true;
		int  case_max_diff = 0;
		for (int i = 0; i < TB_HIDDEN_SIZE; i++) {
			int diff = abs((int)pre_out[i] - TB_PRE_EXPECTED[c][i]);
			if (diff > case_max_diff) case_max_diff = diff;
			if (diff != 0) case_ok = false;
		}
		if (case_max_diff > pre_max_diff) pre_max_diff = case_max_diff;
		if (!case_ok) {
			pre_fail++;
			cout << "HOST-Info:   [PRE FAIL] case=" << c << " layer=" << layer_idx
			     << " max_abs_diff=" << case_max_diff << endl;
		}
	}
	cout << "HOST-Info: PRE  " << (NB_PRE_CASES - pre_fail) << "/" << NB_PRE_CASES
	     << " PASS, max_abs_diff=" << pre_max_diff << endl;

	cout << endl << "HOST-Info: ======================= Running POST cases ======================" << endl;
	int post_fail = 0, post_max_diff = 0;
	for (int c = 0; c < NB_POST_CASES; c++) {
		int layer_idx = TB_POST_LAYER_IDX[c];
		for (int i = 0; i < TB_HIDDEN_SIZE; i++) post_in[i] = (int32_t)TB_POST_INPUT[c][i];

		clSetKernelArg(K_post, 1, sizeof(int), &layer_idx);
		errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &Buf_post_in, 0, 0, NULL, NULL);
		errCode = clEnqueueTask(Command_Queue, K_post, 0, NULL, NULL);
		errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &Buf_post_out, CL_MIGRATE_MEM_OBJECT_HOST, 0, NULL, NULL);
		clFinish(Command_Queue);

		bool case_ok = true;
		int  case_max_diff = 0;
		for (int i = 0; i < TB_HIDDEN_SIZE; i++) {
			int diff = abs((int)post_out[i] - TB_POST_EXPECTED[c][i]);
			if (diff > case_max_diff) case_max_diff = diff;
			if (diff != 0) case_ok = false;
		}
		if (case_max_diff > post_max_diff) post_max_diff = case_max_diff;
		if (!case_ok) {
			post_fail++;
			cout << "HOST-Info:   [POST FAIL] case=" << c << " layer=" << layer_idx
			     << " max_abs_diff=" << case_max_diff << endl;
		}
	}
	cout << "HOST-Info: POST " << (NB_POST_CASES - post_fail) << "/" << NB_POST_CASES
	     << " PASS, max_abs_diff=" << post_max_diff << endl;

	// ===========================================================================
	// Step 6: Verdict + cleanup
	// ===========================================================================
	bool all_pass = (pre_fail == 0) && (post_fail == 0);
	cout << endl << "HOST-Info: =========================================================" << endl;
	cout << (all_pass ? "HOST-Info: Test Successful (hardware matches bit-accurate model)"
	                   : "HOST-Error: Test Failed")
	     << endl << endl;

	clReleaseMemObject(Buf_pre_in);  clReleaseMemObject(Buf_pre_out);
	clReleaseMemObject(Buf_post_in); clReleaseMemObject(Buf_post_out);
	clReleaseKernel(K_pre); clReleaseKernel(K_post);
	clReleaseProgram(Program);
	clReleaseCommandQueue(Command_Queue);
	clReleaseContext(Context);
	delete[] Platform_IDs; delete[] Device_IDs;
	free(pre_in); free(pre_out); free(post_in); free(post_out);

	return all_pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
