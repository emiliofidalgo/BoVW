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

#include "BoVW/bow_trainer.h"

namespace bovw {

BoVWTrainer::BoVWTrainer() :
    nimages_(0),
    ndescriptors_(0)
    {}

BoVWTrainer::~BoVWTrainer() {}

void BoVWTrainer::add(const unsigned img_id, const cv::Mat& descriptors) {
    assert(descriptors_.count(img_id) == 0);

    descriptors_[img_id] = descriptors;
    nimages_++;
    ndescriptors_ += descriptors.rows;
}

void BoVWTrainer::clear() {
    descriptors_.clear();
    nimages_ = 0;
    ndescriptors_ = 0;
}

unsigned BoVWTrainer::numImages() {
    return nimages_;
}

unsigned BoVWTrainer::numDescriptors() {
    return ndescriptors_;
}

void BoVWTrainer::train(
        cv::Mat& vwords,
        cv::Mat& idf,
        const int cluster_count,
        const cv::TermCriteria& termcrit,
        const int attempts,
        const int flags) {
    assert(ndescriptors_ > clusterCount);

    // Merging descriptors in just one cv::Mat and assigning descriptors to images
    cv::Mat descs_merged(ndescriptors_, descriptors_[0].cols, descriptors_[0].type());
    cv::Mat descs_to_img(ndescriptors_, 1, CV_32S);

    int start_copy_idx = 0;
    for (auto img : descriptors_) {
        unsigned img_id = img.first;
        cv::Mat descs = img.second;
        
        // Copying the current set of descriptors
        cv::Mat submat = descs_merged.rowRange(start_copy_idx, (int)(start_copy_idx + descs.rows));
        descs.copyTo(submat);

        // Associating each descriptor to its original image
        submat = descs_to_img.rowRange(start_copy_idx, (int)(start_copy_idx + descs.rows));
        submat.setTo(static_cast<int>(img_id));

        // Updating starting index for copy descriptors
        start_copy_idx += descs.rows;
    }

    // Clustering the set of descriptors
    cv::Mat clusters, labels;
    clusterKMeans(descs_merged, clusters, labels, cluster_count, termcrit, attempts, flags);

    // Copying the resulting dictionary
    clusters.copyTo(vwords);

    // Computing the IDF part
    std::unordered_map<int, std::unordered_set<int> > vwords_apps;
    for (unsigned i = 0; i < ndescriptors_; i++) {
        int vword = labels.at<int>(i);
        int img_id = descs_to_img.at<int>(i);

        if (vwords_apps.count(vword)) {
            // Adding this image to the list of images where the vw was seen
            vwords_apps[vword].insert(img_id);
            
        } else {
            // Adding a new visual word to the result
            std::unordered_set<int> st;
            st.insert(img_id);

            vwords_apps[vword] = st;
        }
    }

    // Filling the idf structure
    idf = cv::Mat::zeros(clusters.rows, 1, CV_32F);
    for (auto it : vwords_apps) {
        int vword = it.first;
        float nimgs = static_cast<float>(it.second.size());

        idf.at<float>(vword) = log(static_cast<float>(nimages_) / nimgs);
    }
}

void BoVWTrainer::clusterKMeans(const cv::Mat& descriptors, 
                     cv::Mat& clusters,
                     cv::Mat& labels,
                     const int cluster_count,
                     const cv::TermCriteria& termcrit,
                     const int attempts,
                     const int flags) {
    cv::kmeans(descriptors, cluster_count, labels, termcrit, attempts, flags, clusters);
}

}  // namespace bovw