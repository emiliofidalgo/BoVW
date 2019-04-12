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

#ifndef LIB_INCLUDE_BOW_TRAINER_H_
#define LIB_INCLUDE_BOW_TRAINER_H_

#include <opencv2/opencv.hpp>

#include <cassert>
#include <cmath>
#include <unordered_map>
#include <unordered_set>

namespace bovw {

class BoVWTrainer {
 public:
  // Constructors
  explicit BoVWTrainer();
  
  // Destructor
  virtual ~BoVWTrainer();

  // Methods
  void add(const unsigned img_id, const cv::Mat& descriptors);
  void clear();
  unsigned numImages();
  unsigned numDescriptors();
  void train(cv::Mat& vwords,
             cv::Mat& idf,
             const int cluster_count,
             const cv::TermCriteria& termcrit = cv::TermCriteria(),
             const int attempts = 3,
             const int flags=cv::KMEANS_PP_CENTERS);

 private:
  std::unordered_map<unsigned, cv::Mat> descriptors_;
  unsigned nimages_;
  unsigned ndescriptors_;

  // Methods
  void clusterKMeans(const cv::Mat& descriptors, 
                     cv::Mat& clusters,
                     cv::Mat& labels,
                     const int cluster_count,
                     const cv::TermCriteria& termcrit,
                     const int attempts,
                     const int flags);
};

}  // namespace bovw

#endif  // LIB_INCLUDE_BOW_TRAINER_H_
