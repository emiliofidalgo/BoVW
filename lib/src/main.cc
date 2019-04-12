/**
* MIT License
* 
* Copyright (c) 2019 Emilio Garcia-Fidalgo (emilio.garcia@uib.es)
* 
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
* 
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

#include <iostream>

#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

#include "BoVW/bow_trainer.h"

void getFilenames(const std::string& directory,
                  std::vector<std::string>* filenames) {
    using namespace boost::filesystem;

    filenames->clear();
    path dir(directory);

    // Retrieving, sorting and filtering filenames.
    std::vector<path> entries;
    copy(directory_iterator(dir), directory_iterator(), back_inserter(entries));
    sort(entries.begin(), entries.end());
    for (auto it = entries.begin(); it != entries.end(); it++) {
        std::string ext = it->extension().c_str();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        if (ext == ".png" || ext == ".jpg" ||
            ext == ".ppm" || ext == ".jpeg") {
            filenames->push_back(it->string());
        }
    }
}

int main(int argc, char** argv) {
  // Creating feature detector and descriptor
  cv::Ptr<cv::xfeatures2d::SURF> surf = cv::xfeatures2d::SURF::create();
  // surf->setMinHessian(400);

  // Loading image filenames
  std::vector<std::string> filenames;
  getFilenames(argv[1], &filenames);
  unsigned nimages = filenames.size();

  // BoW Trainer
  bovw::BoVWTrainer bow_trainer;

  // Processing the sequence of images
  for (unsigned i = 0; i < 5; i++) {
    // Processing image i
    std::cout << "--- Processing image " << i << std::endl;

    // Loading and describing the image
    cv::Mat img = cv::imread(filenames[i]);
    std::vector<cv::KeyPoint> kps;
    surf->detect(img, kps);
    cv::Mat dscs;
    surf->compute(img, kps, dscs);

    // cv::Mat out;
    // cv::drawKeypoints(img, kps, out);

    // cv::imshow("Out", out);
    // cv::waitKey(5);

    bow_trainer.add(i, dscs);
  }

  // Training
  cv::Mat vwords, idf;
  bow_trainer.train(vwords, idf, 100);

  return 0;
}
