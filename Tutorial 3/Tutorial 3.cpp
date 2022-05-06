#include <iostream>
#include <vector>

#include "Utils.h"

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char **argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;

	for (int i = 1; i < argc; i++)	{
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0;}
	}

	//detect any potential exceptions
	try {
		//Part 1 - Load Image
		CImg<unsigned char> image_input(image_filename.c_str());
		CImgDisplay disp_input(image_input, "input");
		
		//Part 2 - host operations
		//2.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Runinng on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context);

		//2.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "kernels/my_kernels.cl");

		cl::Program program(context, sources);

		//build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		typedef int mytype;

		//Part 3 - memory allocation
		//host - input
		std::vector<mytype> A{1,2,3,4,5,5,6,7,8,13};//allocate 10 elements with an initial value 1 - their sum is 10 so it should be easy to check the results!
		
		// Set the amount of work groups to match the amount of available compute units to maximise the amount of code being executed by a unit
		// Using the max compute units and half of the compute units throws an error therefore using a quarter of available units will be done
		// Current compute max compute units is around 4000, therefore 1/4 will be 1000 which is roughly enough compute units for
		// 4 channels of an image 255*4
		int availableComputeUnits = CL_DEVICE_MAX_COMPUTE_UNITS/4;
		size_t local_size = 10;

		size_t padding_size = A.size() % local_size;

		//if the input vector is not a multiple of the local_size
		//insert additional neutral elements (0 for addition) so that the total will not be affected
		if (padding_size) {
			//create an extra vector with neutral values
			std::vector<int> A_ext(local_size-padding_size, 0);
			//append that extra vector to our input
			A.insert(A.end(), A_ext.begin(), A_ext.end());
		}

		size_t input_elements = A.size();//number of input elements
		size_t input_size = A.size()*sizeof(mytype);//size in bytes
		size_t nr_groups = input_elements / local_size;

		//host - output
		std::vector<mytype> B(1);
		size_t output_size = B.size()*sizeof(mytype);//size in bytes

		std::vector<mytype> C(10, 0);
		size_t sizeC = C.size() * sizeof(mytype);//size in bytes
		
		std::vector<mytype> D(10, 0);
		size_t sizeD = D.size() * sizeof(mytype);//size in bytes


		//device - buffers
		cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size);
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_C(context, CL_MEM_READ_ONLY, input_size);
		cl::Buffer buffer_D(context, CL_MEM_READ_WRITE, sizeD);

		//Part 4 - device operations
		
		// Record time
		cl::Event A_event;

		//4.1 copy array A to and initialise other arrays on device memory
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);
		queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);//zero B buffer on device memory

		//4.2 Setup and execute all kernels (i.e. device code)
		//cl::Kernel kernel_1 = cl::Kernel(program, "reduceAdd");
		// Set input
		//kernel_1.setArg(0, buffer_A);
		// Set output
		//kernel_1.setArg(1, buffer_B);
		// Set local memory size
		//kernel_1.setArg(2, cl::Local(local_size*sizeof(mytype)));


		cl::Kernel kernel_2 = cl::Kernel(program, "hist_simple");
		// Set input
		kernel_2.setArg(0, buffer_A);
		kernel_2.setArg(1, buffer_D);
		// Set local memory size
		kernel_2.setArg(2, cl::Local(local_size*sizeof(mytype)));

		//call all kernels in a sequence
		//queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &A_event);
		queue.enqueueNDRangeKernel(kernel_2, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &A_event);


		//4.3 Copy the result from device to host
		queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]);
		queue.enqueueReadBuffer(buffer_D, CL_TRUE, 0, sizeC, &D[0]);


		std::cout << "A = " << A << std::endl;
		std::cout << "B = " << B << std::endl;
		std::cout << "C = " << C << std::endl;
		std::cout << "D = " << D << std::endl;

		std::cout << "Preferred WG Size" << CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE << std::endl;
		std::cout << "Actual WG Size" << availableComputeUnits << std::endl;

		// stuff for images 
		// 
		//device - buffers
		cl::Buffer dev_image_input(context, CL_MEM_READ_ONLY, image_input.size());
		cl::Buffer dev_image_output(context, CL_MEM_READ_WRITE, image_input.size()); //should be the same as input image
//		cl::Buffer dev_convolution_mask(context, CL_MEM_READ_ONLY, convolution_mask.size()*sizeof(float));

		//4.1 Copy images to device memory
		queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0]);
		//		queue.enqueueWriteBuffer(dev_convolution_mask, CL_TRUE, 0, convolution_mask.size()*sizeof(float), &convolution_mask[0]);
		cl::Kernel kernel_filter_r = cl::Kernel(program, "filter_r");
		kernel_filter_r.setArg(0, dev_image_input);
		kernel_filter_r.setArg(1, dev_image_output);


		//4.2 Setup and execute the kernel (i.e. device code)
		/*cl::Kernel kernel = cl::Kernel(program, "identity");
		kernel.setArg(0, dev_image_input);
		kernel.setArg(1, dev_image_output);*/
		//		kernel.setArg(2, dev_convolution_mask);

		queue.enqueueNDRangeKernel(kernel_filter_r, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange);

		vector<unsigned char> output_buffer(image_input.size());
		//4.3 Copy the result from device to host
		queue.enqueueReadBuffer(dev_image_output, CL_TRUE, 0, output_buffer.size(), &output_buffer.data()[0]);

		CImg<unsigned char> output_image(output_buffer.data(), image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum());
		CImgDisplay disp_output(output_image, "output");

		while (!disp_input.is_closed() && !disp_output.is_closed()
			&& !disp_input.is_keyESC() && !disp_output.is_keyESC()) {
			disp_input.wait(1);
			disp_output.wait(1);
		}


	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	catch (const cl::Error& err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	catch (CImgException& err) {
		std::cerr << "ERROR: " << err.what() << std::endl;
	}


	return 0;
}


kernel void identity(global const uchar* A, global uchar* B) {
	int id = get_global_id(0);
	B[id] = A[id];
}

kernel void filter_r(global const uchar* A, global uchar* B) {
	int id = get_global_id(0);
	int image_size = get_global_size(0) / 3; //each image consists of 3 colour channels
	int colour_channel = id / image_size; // 0 - red, 1 - green, 2 - blue

	//this is just a copy operation, modify to filter out the individual colour channels
	if (colour_channel == 0) {
		B[id] = A[id];
	}

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