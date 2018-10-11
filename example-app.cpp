#include <iostream>

#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace cv;

void showImage(Mat image)
{
    namedWindow( "Display window", CV_WINDOW_AUTOSIZE );// Create a window for display.
    imshow("Display window", image );
    waitKey(0);
}


int main(int argc, const char* argv[]) {
    //std::cout << "OpenCV version : " << CV_VERSION << std::endl;
	Mat image;
	image = imread("../testSample/img_63.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	showImage(image);
	// Deserialize the ScriptModule from a file using torch::jit::load().
	std::shared_ptr<torch::jit::script::Module> module = torch::jit::load("../model_trace.pt");
	std::vector<int64_t> sizes = {1, 1, image.rows, image.cols};
	at::TensorOptions options(at::ScalarType::Byte);
    at::Tensor tensor_image = torch::from_blob(image.data, at::IntList(sizes), options);
    tensor_image = tensor_image.toType(at::kFloat);
    at::Tensor result = module->forward({tensor_image}).toTensor();

    auto max_result = result.max(1, true);
    auto max_index = std::get<1>(max_result).item<float>();
    std::cout << max_index << std::endl;

}