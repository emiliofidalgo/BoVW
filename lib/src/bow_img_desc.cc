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

#include "BoVW/bow_img_desc.h"

namespace bovw {

BoVWImageDescriptor::BoVWImageDescriptor(const cv::Ptr<cv::DescriptorMatcher>& _dmatcher ) :
    dmatcher(_dmatcher)
{}

BoVWImageDescriptor::~BoVWImageDescriptor() {}

void BoVWImageDescriptor::setVocabulary( const cv::Mat _vocabulary )
{
    dmatcher->clear();
    vocabulary = _vocabulary;
    dmatcher->add( std::vector<cv::Mat>(1, vocabulary) );
}

const cv::Mat& BoVWImageDescriptor::getVocabulary() const
{
    return vocabulary;
}

int BoVWImageDescriptor::descriptorSize() const
{
    return vocabulary.empty() ? 0 : vocabulary.rows;
}

int BoVWImageDescriptor::descriptorType() const
{
    return CV_32FC1;
}

void BoVWImageDescriptor::compute( 
                const cv::Mat& descriptors, 
                 cv::Mat& desc_clusters, 
                 cv::Mat& apps, 
                 cv::Mat& normalized_bovw_desc)
{
    CV_Assert(!vocabulary.empty() );
    CV_Assert(!descriptors.empty());

    int clusterCount = descriptorSize(); // = vocabulary.rows

    desc_clusters = cv::Mat::zeros(descriptors.rows, 1, CV_32S);
    apps = cv::Mat::zeros(1, clusterCount, CV_32S);

    // Match descriptors to cluster center (to vocabulary)
    std::vector<cv::DMatch> matches;
    dmatcher->match( descriptors, matches );
    
    for(size_t i=0; i<matches.size(); i++)
    {
        int queryIdx = matches[i].queryIdx;
        int trainIdx = matches[i].trainIdx; // cluster index
        CV_Assert( queryIdx == (int)i );
        
        // Updating cluster assignment
        desc_clusters.at<int>(queryIdx,0)= trainIdx;
        
        // Updating number of appearances
        apps.at<int>(0, trainIdx) += 1;
    }

    apps.convertTo(normalized_bovw_desc, CV_32F);
    normalized_bovw_desc /= descriptors.rows;
}

}