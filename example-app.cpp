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
  // 转化格式，opencv读取的图像矩阵存储形式：H x W x C, 
  // 但是pytorch中 Tensor的存储为：N x C x H x W, 因此需要进行变换，就是np.transpose()操作，这里使用tensor.permut()实现，效果是一样的。
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
  std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/10) << '\n';

  // Load labels
  std::string label_file = argv[3];
  std::ifstream rf(label_file.c_str());
  CHECK(rf) << "Unable to open file " << label_file;
  std::string line;
  std::vector<std::string> labels;
  while (std::getline(rf, line))
    labels.push_back(line);

  // print predicted top-5 labels
  std::tuple<torch::Tensor,torch::Tensor> result = output.sort(-1, true);
  torch::Tensor top_scores = std::get<0>(result)[0];
  torch::Tensor top_idxs = std::get<1>(result)[0].toType(torch::kInt32);
  
  auto top_scores_a = top_scores.accessor<float,1>();
  auto top_idxs_a = top_idxs.accessor<int,1>();

  for (int i = 0; i < 5; ++i) {
    int idx = top_idxs_a[i];
    std::cout << "top-" << i+1 << " label: ";
    std::cout << labels[idx] << ", score: " << top_scores_a[i] << std::endl;
  }
}