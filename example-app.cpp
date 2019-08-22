#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

#include <opencv2/opencv.hpp>
using namespace cv;

int main(int argc, const char* argv[]) {
  // if (argc != 2) {
  //   std::cerr << "usage: example-app <path-to-exported-script-module>\n";
  //   return -1;
  // }


  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }
  
  cv::Mat image;
  image = cv::imread(argv[2], 1);
  cv::cvtColor(image, image, CV_BGR2RGB);
  cv::Mat img_float;
  image.convertTo(img_float, CV_32F, 1.0/255);
  cv::resize(img_float, img_float, cv::Size(224, 224));
  //std::cout << img_float.at<cv::Vec3f>(56,34)[1] << std::endl;
  auto img_tensor = torch::from_blob(img_float.data, {1, 224, 224, 3});
  // torch::Tensor img_tensor = batch.data.to(device);
  img_tensor = img_tensor.permute({0,3,1,2});
  img_tensor[0][0] = img_tensor[0][0].sub_(0.485).div_(0.229);
  img_tensor[0][1] = img_tensor[0][1].sub_(0.456).div_(0.224);
  img_tensor[0][2] = img_tensor[0][2].sub_(0.406).div_(0.225);

  std::cout << "ok\n";
  // Create a vector of inputs.
  std::vector<torch::jit::IValue> inputs;
  // C++ 读取图片，转换成torch输入格式
  // inputs.push_back(torch::ones({1, 3, 224, 224}));
  // inputs.push_back(torch::ones({1, 3, 224, 224}));
  inputs.push_back(img_tensor);

  // Execute the model and turn its output into a tensor.
  at::Tensor output = module.forward(inputs).toTensor();
  std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
}