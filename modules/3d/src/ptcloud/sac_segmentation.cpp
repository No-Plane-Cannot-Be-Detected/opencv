// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.


#include "../precomp.hpp"
//#include "sac_segmentation.hpp"
#include "opencv2/3d/ptcloud.hpp"
#include "ptcloud_utils.hpp"
#include "../usac.hpp"

namespace cv {
//    namespace _3d {

/////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////  SACSegmentation  ////////////////////////////////////////


void SACSegmentation::setPointCloud(const Mat &input_pts_)
{
    input_pts = input_pts_;
}

//-------------------------- segment -----------------------
int SACSegmentation::segment(OutputArray labels, OutputArray models_coefficients)
{

    // Since error function output squared error distance, so make
    // threshold squared as well
    double thr = threshold * threshold;

    // RANSAC
    Ptr <usac::Model> param = usac::Model::create(
            thr, usac::EstimationMethod::POINT_CLOUD_MODEL, SamplingMethod::SAMPLING_UNIFORM,
            probability, max_iterations, ScoreMethod::SCORE_METHOD_RANSAC);

    param->setPolisher(usac::NonePolisher);
    param->setSampleSize(3);


    std::vector<Mat> models_coeffs;

    Mat pts = input_pts;
    int pts_size = pts.rows;
    std::vector<int> _labels(pts_size, 0);


    // Keep the index array of the point corresponding to the original point
    AutoBuffer<int> ori_pts_idx(pts_size);
    int *pts_idx_ptr = ori_pts_idx.data();
    for (int i = 0; i < pts_size; ++i) pts_idx_ptr[i] = i;

    for (int model_num = 1; model_num <= number_of_models_expected; ++model_num)
    {
        Ptr <usac::RansacOutput> ransacOutput;
        if (pts.empty() || !usac::fittingGeometricModelBySAC(param, pts, ransacOutput))
            break;

        models_coeffs.emplace_back(ransacOutput->getModel());

        std::vector<bool> mask = ransacOutput->getInliersMask();

        pts_size = pts.rows;
        if (model_num != number_of_models_expected)
        {
            int best_inls = ransacOutput->getNumberOfInliers();
            cv::Mat tmp_pts(pts);
            pts = cv::Mat(pts_size - best_inls, 3, CV_32F);

            float *const tmp_pts_ptr = (float *) tmp_pts.data;
            float *const pts_ptr = (float *) pts.data;
            for (int j = 0, k = 0; k < pts_size; ++k)
            {
                if (mask[k])
                {
                    // mark a label on this point
                    _labels[pts_idx_ptr[k]] = model_num;
                }
                else
                {
                    // If it is not inlier of the known plane,
                    //   add the next iteration to find a new plane
                    pts_idx_ptr[j] = pts_idx_ptr[k];
                    float *const tmp_ptr_base = tmp_pts_ptr + 3 * k;
                    float *const pts_fit_ptr_base = pts_ptr + 3 * j;
                    pts_fit_ptr_base[0] = tmp_ptr_base[0];
                    pts_fit_ptr_base[1] = tmp_ptr_base[1];
                    pts_fit_ptr_base[2] = tmp_ptr_base[2];
                    ++j;
                }
            }
        }
        else
        {
            for (int k = 0; k < pts_size; ++k)
            {
                if (mask[k])
                    _labels[pts_idx_ptr[k]] = model_num;
            }
        }
    }

    int number_of_models = (int) models_coeffs.size();
    if (labels.needed())
    {
        if (number_of_models != 0)
        {
            Mat(_labels).copyTo(labels);
        }
        else
        {
            labels.clear();
        }
    }


    //    Mat(models_coeffs).copyTo(models_coefficients);
    if (models_coefficients.needed())
    {
        if (number_of_models != 0)
        {

            models_coefficients.create(number_of_models, 1, models_coeffs[0].type());
            /**get vector*/
            std::vector<Mat> dst;
            models_coefficients.getMatVector(dst);
            for (int i = 0; i < number_of_models; i++)
            {
                Mat cur_mat = models_coeffs[i];
                models_coefficients.getMatRef(i) = cur_mat;
            }
        }
        else
        {
            models_coefficients.clear();
        }

    }

    return number_of_models;
}

//    } // _3d::
}  // cv::