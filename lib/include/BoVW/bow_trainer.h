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

namespace bovw {

class BoVWTrainer {
 public:
  // Constructors
  explicit BoVWTrainer();
  
  // Destructor
  virtual ~BoVWTrainer();

 private:
  std::vector<cv::Mat> descriptors_;
  unsigned total_size_;
};

}  // namespace bovw

#endif  // LIB_INCLUDE_BOW_TRAINER_H_
