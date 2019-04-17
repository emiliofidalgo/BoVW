/**
* MIT License
* 
* Copyright (c) 2019 Rihem El Euch (rihem.eleuch@supcom.tn)
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

#ifndef LIB_INCLUDE_BOW_DESCRIPTOR_H_
#define LIB_INCLUDE_BOW_DESCRIPTOR_H_

#include <opencv2/opencv.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/core.hpp"

#include "opencv2/highgui/highgui.hpp" 

using namespace cv::xfeatures2d;

namespace bovw {

class BoVWImageDescriptor {

public:
   BoVWImageDescriptor ( const cv::Ptr<cv::DescriptorMatcher>& dmatcher );
   virtual ~BoVWImageDescriptor();

   //methods
   void setVocabulary( const cv::Mat vocabulary );
   const cv::Mat& getVocabulary() const;
   void compute( const cv::Mat& descriptors, 
                 cv::Mat& desc_clusters, 
                 cv::Mat& apps,
                 cv::Mat& normalized_bovw_desc);

   int descriptorSize() const;
   int descriptorType() const;

protected:
    cv::Mat vocabulary;
    cv::Ptr<cv::DescriptorMatcher> dmatcher;
};

}
#endif  // LIB_INCLUDE_BOW_DESCRIPTOR_H_