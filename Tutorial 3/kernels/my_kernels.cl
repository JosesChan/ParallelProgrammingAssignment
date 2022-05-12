// flexible step reduce using local memory instead of global,
// reduce using local memory (Think about shared memory?)
// 
kernel void reduceAdd(global const int* A, global int* B, local int* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			scratch[lid] += scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	////we add results from all local groups to the first element of the array
	////serial operation! but works for any group size
	////copy the cache to output array
	//if (!lid) {
	//	atomic_add(&B[0],scratch[lid]);
	//}

	B[id] = scratch[lid];
}

kernel void histSimpleImplement(global const uchar* A, global int* H) {
	
	size_t globalID = get_global_id(0);
	
	//assumes that H has been initialised to 0
	int bin_index = A[globalID];//take value as a bin index

	atomic_inc(&H[bin_index]);//serial operation, not very efficient!
}

//kernel void histLocalSimple(global const uchar* A, global int* H, local int* LH, int nr_bins) {
//
//	size_t globalID = get_global_id(0);
//	size_t localID = get_local_id(0);
//	size_t bin_index = A[id];
//
//	//assumes that H has been initialised to 0
//	int bin_index = A[globalID];//take value as a bin index
//
//	atomic_inc(&H[bin_index]);//serial operation, not very efficient!
//}
//
//kernel void histLocalPrototype(global const uchar* A, global int* H, local int* LH, int nr_bins) {
//
//	size_t globalID = get_global_id(0);
//	size_t localID = get_local_id(0);
//	size_t bin_index = A[id];
//
//	//assumes that H has been initialised to 0
//	int bin_index = A[globalID];//take value as a bin index
//
//	atomic_inc(&H[bin_index]);//serial operation, not very efficient!
//}

kernel void histLocalSimple(global const uchar* A, global int* H, local int* LH, int nr_bins) {
	size_t globalID = get_global_id(0);
	size_t localID = get_local_id(0);
	int bin_index = A[globalID];
	//assumes that H has been initialised to 0
	//int bin_index = A[globalID];//take value as a bin index

	// set bin to 0
	if (localID < nr_bins) {
		LH[localID] = 0;
	}

	barrier(CLK_LOCAL_MEM_FENCE);
	atomic_inc(&LH[bin_index]);
	barrier(CLK_LOCAL_MEM_FENCE);

	// sync then combine privatised histograms
	barrier(CLK_LOCAL_MEM_FENCE);
	atomic_add(&H[localID], LH[localID]);
}


// kernel to look at colour histograms
# define BIN_SIZE 256
void colour_histogram_kernel(global const uint* data, global uint* binResultR, global uint* binResultG, global uint* binResultB, int elements_awaiting_process, int total_pixels) {
	__global uchar4* image_data = data;
	__local int sharedArrayR[BIN_SIZE];
	__local int sharedArrayG[BIN_SIZE];
	__local int sharedArrayB[BIN_SIZE];
	size_t localID = get_local_id(0);
	size_t globalID = get_global_id(0);
	size_t groupID = get_group_id(0);
	size_t groupSize = get_local_size(0);
	
	// for coalesced access, smaller size then a naive implementation
	// set only once by each work item
	sharedArrayR[localID] = 0;
	sharedArrayG[localID] = 0;
	sharedArrayB[localID] = 0;
	// synchronise setting to 0
	barrier(CLK_LOCAL_MEM_FENCE);

	int groupOffset = groupID * groupSize * elements_awaiting_process;

	// Calculate the number of elements required for last work group
	if (groupID == (get_num_groups(0) - 1)) 
		elements_awaiting_process = ((total_pixels - groupOffset) + groupSize - 1) / groupSize;
	
	// calculate thread histogram
	for (int i = 0; i < elements_awaiting_process; ++i) {
		int index = groupOffset + i * get_local_size(0) + localID;
		// ensure boundary conditions (global mem not outside global mem buffer range)
		if (index > total_pixels)
			break;

		// memory coalesced from global mem, combining multiple memory accesses into single transaction later
		// using atomic inc to increment reading pixel values from global memory
		uchar4 value = image_data[index];
		atomic_inc(&sharedArrayR[value.x]);
		atomic_inc(&sharedArrayR[value.y]);
		atomic_inc(&sharedArrayR[value.w]);
	}

	// wait for all work items to process awaiting elements
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// memory coalesced, written to global memory to combine
	binResultR[groupID * BIN_SIZE + localID] = sharedArrayR[localID];
	binResultG[groupID * BIN_SIZE + localID] = sharedArrayR[localID];
	binResultB[groupID * BIN_SIZE + localID] = sharedArrayR[localID];
}


//a very simple histogram implementation
kernel void hist_simple(global const int* A, global int* H, local int* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	//assumes that H has been initialised to 0
	int bin_index = A[lid];//take value as a bin index

	barrier(CLK_LOCAL_MEM_FENCE);

	atomic_inc(&H[bin_index]);//serial operation, not very efficient!

	barrier(CLK_LOCAL_MEM_FENCE);

}

//Hillis-Steele basic inclusive scan
//requires additional buffer B to avoid data overwrite 
kernel void scan_hs(global int* A, global int* B) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	global int* C;

	for (int stride = 1; stride < N; stride *= 2) {
		B[id] = A[id];
		if (id >= stride)
			B[id] += A[id - stride];

		barrier(CLK_GLOBAL_MEM_FENCE); //sync the step

		C = A; A = B; B = C; //swap A & B between steps
	}
}

//a double-buffered version of the Hillis-Steele inclusive scan
//requires two additional input arguments which correspond to two local buffers
kernel void scan_add(__global const int* A, global int* B, local int* scratch_1, local int* scratch_2) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);
	local int *scratch_3;//used for buffer swap

	//cache all N values from global memory to local memory
	scratch_1[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) {
		if (lid >= i)
			scratch_2[lid] = scratch_1[lid] + scratch_1[lid - i];
		else
			scratch_2[lid] = scratch_1[lid];

		barrier(CLK_LOCAL_MEM_FENCE);

		//buffer swap
		scratch_3 = scratch_2;
		scratch_2 = scratch_1;
		scratch_1 = scratch_3;
	}

	//copy the cache to output array
	B[id] = scratch_1[lid];
}

//Blelloch basic exclusive scan
kernel void scan_bl(global int* A) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	int t;

	//up-sweep
	for (int stride = 1; stride < N; stride *= 2) {
		if (((id + 1) % (stride*2)) == 0)
			A[id] += A[id - stride];

		barrier(CLK_GLOBAL_MEM_FENCE); //sync the step
	}

	//down-sweep
	if (id == 0)
		A[N-1] = 0;//exclusive scan

	barrier(CLK_GLOBAL_MEM_FENCE); //sync the step

	for (int stride = N/2; stride > 0; stride /= 2) {
		if (((id + 1) % (stride*2)) == 0) {
			t = A[id];
			A[id] += A[id - stride]; //reduce 
			A[id - stride] = t;		 //move
		}

		barrier(CLK_GLOBAL_MEM_FENCE); //sync the step
	}
}

//calculates the block sums
kernel void block_sum(global const int* A, global int* B, int local_size) {
	int id = get_global_id(0);
	B[id] = A[(id+1)*local_size-1];
}

//simple exclusive serial scan based on atomic operations - sufficient for small number of elements
kernel void scan_add_atomic(global int* A, global int* B) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	for (int i = id+1; i < N && id < N; i++)
		atomic_add(&B[i], A[id]);
}

//adjust the values stored in partial scans by adding block sums to corresponding blocks
kernel void scan_add_adjust(global int* A, global const int* B) {
	int id = get_global_id(0);
	int gid = get_group_id(0);
	A[id] += B[gid];
}

kernel void identity(global const uchar* A, global uchar* B) {
	int id = get_global_id(0);
	B[id] = A[id];
}

//simple ND identity kernel
kernel void identityND(global const uchar* A, global uchar* B) {
	int width = get_global_size(0); //image width in pixels
	int height = get_global_size(1); //image height in pixels
	int image_size = width * height; //image size in pixels
	int channels = get_global_size(2); //number of colour channels: 3 for RGB

	int x = get_global_id(0); //current x coord.
	int y = get_global_id(1); //current y coord.
	int c = get_global_id(2); //current colour channel

	int id = x + y * width + c * image_size; //global id in 1D space

	B[id] = A[id];
}

//2D averaging filter
kernel void avg_filterND(global const uchar* A, global uchar* B) {
	int width = get_global_size(0); //image width in pixels
	int height = get_global_size(1); //image height in pixels
	int image_size = width * height; //image size in pixels
	int channels = get_global_size(2); //number of colour channels: 3 for RGB

	int x = get_global_id(0); //current x coord.
	int y = get_global_id(1); //current y coord.
	int c = get_global_id(2); //current colour channel

	int id = x + y * width + c * image_size; //global id in 1D space

	uint result = 0;

	//simple boundary handling - just copy the original pixel
	if ((x == 0) || (x == width - 1) || (y == 0) || (y == height - 1)) {
		result = A[id];
	}
	else {
		for (int i = (x - 1); i <= (x + 1); i++)
			for (int j = (y - 1); j <= (y + 1); j++)
				result += A[i + j * width + c * image_size];

		result /= 9;
	}

	B[id] = (uchar)result;
}

//2D 3x3 convolution kernel
kernel void convolutionND(global const uchar* A, global uchar* B, constant float* mask) {
	int width = get_global_size(0); //image width in pixels
	int height = get_global_size(1); //image height in pixels
	int image_size = width * height; //image size in pixels
	int channels = get_global_size(2); //number of colour channels: 3 for RGB

	int x = get_global_id(0); //current x coord.
	int y = get_global_id(1); //current y coord.
	int c = get_global_id(2); //current colour channel

	int id = x + y * width + c * image_size; //global id in 1D space

	float result = 0;

	//simple boundary handling - just copy the original pixel
	if ((x == 0) || (x == width - 1) || (y == 0) || (y == height - 1)) {
		result = A[id];
	}
	else {
		for (int i = (x - 1); i <= (x + 1); i++)
			for (int j = (y - 1); j <= (y + 1); j++)
				result += A[i + j * width + c * image_size] * mask[i - (x - 1) + j - (y - 1)];
	}

	B[id] = (uchar)result;
}

kernel void LUT(global int* cumulativeHistogram, global int* lookupTable) {
	size_t globalID = get_global_id(0);
	lookupTable[globalID] = cumulativeHistogram[globalID] * (double)255 / cumulativeHistogram[255];
}

kernel void backProjection(global uchar* A, global int* lookupTable, global uchar* B) {
	size_t globalID = get_global_id(0);
	B[globalID] = lookupTable[A[globalID]];
}