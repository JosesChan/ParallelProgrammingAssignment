/*
*/

#include <iostream>
#include <vector>

#include "Utils.h"
#include "CImg.h"

using namespace cimg_library;

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
	string image_filename= "test.pgm";

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
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

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
		std::vector<mytype> intensityHistogram(256);
		std::vector<mytype> cumulativeHistogram(256);
		std::vector<mytype> lookUpTable(256);

		// Set the amount of work groups to match the amount of available compute units to maximise the amount of code being executed by a unit
		// Using the max compute units and half of the compute units throws an error therefore using a quarter of available units will be done
		// Current compute max compute units is around 4000, therefore 1/4 will be 1000 which is roughly enough compute units for
		// 4 channels of an image 255*4
		int availableComputeUnits = CL_DEVICE_MAX_COMPUTE_UNITS/4;
		size_t local_size = availableComputeUnits;

		/*size_t padding_size = intensityHistogram.size() % local_size;*/

		//if the input vector is not a multiple of the local_size
		//insert additional neutral elements (0 for addition) so that the total will not be affected
		//if (padding_size) {
		//	// create an extra vector with neutral values
		//	std::vector<int> A_ext(local_size-padding_size, 0);
		//	// append that extra vector to our input
		//	intensityHistogram.insert(intensityHistogram.end(), A_ext.begin(), A_ext.end());
		//}

		size_t input_elements = intensityHistogram.size();//number of input elements
		size_t input_size = intensityHistogram.size()*sizeof(mytype);//size in bytes
		size_t nr_groups = input_elements / local_size;
		size_t elementsInput = image_input.size();

		//host - output
		//std::vector<mytype> B(1);
		//size_t output_size = B.size()*sizeof(mytype);//size in bytes

		//std::vector<mytype> C(10, 0);
		//size_t sizeC = C.size() * sizeof(mytype);//size in bytes
		
		//std::vector<mytype> D(10, 0);
		//size_t sizeD = D.size() * sizeof(mytype);//size in bytes


		//device - buffers
		cl::Buffer dev_image_input(context, CL_MEM_READ_ONLY, image_input.size());
		cl::Buffer dev_image_output(context, CL_MEM_READ_WRITE, image_input.size()); //should be the same as input image
		cl::Buffer bufferIntensityHistogram(context, CL_MEM_READ_WRITE, input_size);
		cl::Buffer bufferCumulativeHistogram(context, CL_MEM_READ_WRITE, input_size);
		cl::Buffer bufferLookUpTable(context, CL_MEM_READ_WRITE, input_size);
		// complex hist required buffers, unimplemented as of this moment
		cl::Buffer intermediateHistR(context, CL_MEM_WRITE_ONLY, input_size);
		cl::Buffer intermediateHistG(context, CL_MEM_WRITE_ONLY, input_size);
		cl::Buffer intermediateHistB(context, CL_MEM_WRITE_ONLY, input_size);


		//Part 4 - device operations
		
		//4.1 copy array A to and initialise other arrays on device memory
		queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0]);
		// set zero values in output buffers
		queue.enqueueFillBuffer(bufferIntensityHistogram, 0, 0, input_size);
		queue.enqueueFillBuffer(bufferCumulativeHistogram, 0, 0, input_size);
		queue.enqueueFillBuffer(bufferLookUpTable, 0, 0, input_size);

		//4.2 Setup and execute all kernels (i.e. device code)

		//colour_histogram_kernel(global const uint * data, global uint * binResultR, global uint * binResultG, global uint * binResultB, int elements_awaiting_process, int total_pixels)
		
		cl::Kernel kernel_1 = cl::Kernel(program, "histLocalSimple");
		// Set input
		kernel_1.setArg(0, dev_image_input);
		// Set output
		kernel_1.setArg(1, bufferIntensityHistogram);
		kernel_1.setArg(2, cl::Local(local_size));
		kernel_1.setArg(3, int(nr_groups));
		
		// unimplemented code below
		//cl::Kernel kernel_1 = cl::Kernel(program, "colour_histogram_kernel");
		//// Set input
		//kernel_1.setArg(0, dev_image_input);
		//kernel_1.setArg(1, intermediateHistR);
		//kernel_1.setArg(2, intermediateHistG));
		//kernel_1.setArg(3, intermediateHistB);
		//kernel_1.setArg(4, cl::Local(local_size));
		//kernel_1.setArg(5, cl::Local(local_size));
		

		cl::Kernel kernel_2 = cl::Kernel(program, "scan_add");
		// Set input
		kernel_2.setArg(0, bufferIntensityHistogram);
		// Set output
		kernel_2.setArg(1, bufferCumulativeHistogram);
		// Allocate local memory
		kernel_2.setArg(2, cl::Local(local_size));
		kernel_2.setArg(3, cl::Local(local_size));
		
		cl::Kernel kernel_3 = cl::Kernel(program, "LUT");
		// Set input
		kernel_3.setArg(0, bufferCumulativeHistogram);
		// Set output
		kernel_3.setArg(1, bufferLookUpTable);

		cl::Kernel kernel_4 = cl::Kernel(program, "backProjection");
		// Set input
		kernel_4.setArg(0, dev_image_input);
		kernel_4.setArg(1, bufferLookUpTable);
		// Set output
		kernel_4.setArg(2, dev_image_output);

		// create vector to store image
		vector<unsigned char> output_image_buffer(image_input.size());

		//call all kernels in a sequence and record time
		cl::Event timeIHist;
		queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &timeIHist);
		queue.enqueueReadBuffer(bufferIntensityHistogram, CL_TRUE, 0, input_size, &intensityHistogram[0]);
		cl::Event timeCumulativeHist;
		queue.enqueueNDRangeKernel(kernel_2, cl::NullRange, cl::NDRange(input_size), cl::NullRange, NULL, &timeCumulativeHist);
		queue.enqueueReadBuffer(bufferCumulativeHistogram, CL_TRUE, 0, input_size, &cumulativeHistogram[0]);
		cl::Event timeLut;
		queue.enqueueNDRangeKernel(kernel_3, cl::NullRange, cl::NDRange(input_size), cl::NullRange, NULL, &timeLut);
		queue.enqueueReadBuffer(bufferLookUpTable, CL_TRUE, 0, input_size, &lookUpTable[0]);
		cl::Event timeProjection;
		queue.enqueueNDRangeKernel(kernel_4, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &timeProjection);
		queue.enqueueReadBuffer(dev_image_output, CL_TRUE, 0, output_image_buffer.size(), &output_image_buffer.data()[0]);

		//4.3 Results
		std::cout << "Intensity Histogram Values : " << intensityHistogram << std::endl;
		std::cout << "lol" << std::endl;
		std::cout << "Histogram kernel execution time [ns]: " << timeIHist.getProfilingInfo<CL_PROFILING_COMMAND_END>() - timeIHist.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << GetFullProfilingInfo(timeIHist, ProfilingResolution::PROF_US) << endl;
		cout << endl;

		cout << endl;
		std::cout << "Cumulative Histogram data = " << cumulativeHistogram << std::endl;
		std::cout << "Cumulative Histogram execute time in nanoseconds : " << timeCumulativeHist.getProfilingInfo<CL_PROFILING_COMMAND_END>() - timeCumulativeHist.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << GetFullProfilingInfo(timeCumulativeHist, ProfilingResolution::PROF_US) << endl;
		cout << endl;

		cout << endl;
		std::cout << "Look-up table data = " << lookUpTable << std::endl;
		std::cout << "Look-up table execute time in nanoseconds : " << timeLut.getProfilingInfo<CL_PROFILING_COMMAND_END>() - timeLut.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << GetFullProfilingInfo(timeLut, ProfilingResolution::PROF_US) << endl;
		cout << endl;

		cout << endl;
		std::cout << "Vector kernel execute time in nanoseconds : " << timeProjection.getProfilingInfo<CL_PROFILING_COMMAND_END>() - timeProjection.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << GetFullProfilingInfo(timeProjection, ProfilingResolution::PROF_US) << endl;
		cout << endl;

		cout << endl;
		std::cout << "Preferred WG Size" << CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE << std::endl;
		std::cout << "Actual WG Size" << availableComputeUnits << std::endl;
		cout << endl;

		std::cout << "Image Size = "<< elementsInput  << std::endl;
		
		CImg<unsigned char> output_image(output_image_buffer.data(), image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum());
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
	catch (CImgException& err) {
		std::cerr << "ERROR: " << err.what() << std::endl;
	}
	return 0;
}