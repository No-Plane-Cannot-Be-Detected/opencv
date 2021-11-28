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


int
SACSegmentation::segmentSingle(Mat &points, std::vector<bool> &label, Mat &model_coefficients)
{
    // Since error function output squared error distance, so make
    // threshold squared as well
    double _threshold = threshold * threshold;
    int state = (int) rng_state;
    const int points_size = points.rows * points.cols / 3;

    // RANSAC
    using namespace usac;
    int _max_iterations_before_lo = 100, _max_num_hypothesis_to_test_before_rejection = 15;
    SamplingMethod _sampling_method = SamplingMethod::SAMPLING_UNIFORM;
    LocalOptimMethod _lo_method = LocalOptimMethod::LOCAL_OPTIM_INNER_LO;
    ScoreMethod _score_method = ScoreMethod::SCORE_METHOD_RANSAC;
    NeighborSearchMethod _neighbors_search_method = NeighborSearchMethod::NEIGH_GRID;

    // Local optimization
    int lo_sample_size = 16, lo_inner_iterations = 15, lo_iterative_iterations = 8,
            lo_thr_multiplier = 15, lo_iter_sample_size = 30;

    Ptr <Sampler> sampler;
    Ptr <Quality> quality;
    Ptr <ModelVerifier> verifier;
    Ptr <LocalOptimization> lo;
    Ptr <Degeneracy> degeneracy;
    Ptr <TerminationCriteria> termination;
    Ptr <FinalModelPolisher> polisher;
    Ptr <MinimalSolver> min_solver;
    Ptr <NonMinimalSolver> non_min_solver;
    Ptr <Estimator> estimator;
    Ptr <usac::Error> error;

    switch (sac_model_type)
    {
        case SAC_MODEL_PLANE:
            min_solver = PlaneModelMinimalSolver::create(points);
            non_min_solver = PlaneModelNonMinimalSolver::create(points);
            error = PlaneModelError::create(points);
            break;
//        case SAC_MODEL_CYLINDER:
//            min_solver = CylinderModelMinimalSolver::create(points);
//            non_min_solver = CylinderModelNonMinimalSolver::create(points);
//            error = CylinderModelError::create(points);
//            break;
        case SAC_MODEL_SPHERE:
            min_solver = SphereModelMinimalSolver::create(points);
            non_min_solver = SphereModelNonMinimalSolver::create(points);
            error = SphereModelError::create(points);
            break;
        default:
            CV_Error(cv::Error::StsNotImplemented, "SAC_MODEL type is not implemented!");
    }

    const int min_sample_size = min_solver->getSampleSize();

    if (points_size < min_sample_size)
    {
        return 0;
    }

    estimator = PointCloudModelEstimator::create(min_solver, non_min_solver);
    sampler = UniformSampler::create(state++, min_sample_size, points_size);
    quality = RansacQuality::create(points_size, _threshold, error);
    verifier = ModelVerifier::create();


    Ptr <RandomGenerator> lo_sampler = UniformRandomGenerator::create(state++, points_size,
            lo_sample_size);

    lo = InnerIterativeLocalOptimization::create(estimator, quality, lo_sampler, points_size,
            _threshold, false, lo_iter_sample_size, lo_inner_iterations,
            lo_iterative_iterations, lo_thr_multiplier);

    degeneracy = makePtr<Degeneracy>();
    termination = StandardTerminationCriteria::create
            (probability, points_size, min_sample_size, max_iterations);

    Ptr <SimpleUsacConfig> usacConfig = SimpleUsacConfig::create();
    usacConfig->setThreshold(_threshold);
    usacConfig->setMaxIterations(max_iterations);
    usacConfig->setMaxIterationsBeforeLo(_max_iterations_before_lo);
    usacConfig->setMaxNumHypothesisToTestBeforeRejection(
            _max_num_hypothesis_to_test_before_rejection);
    usacConfig->setConfidence(probability);
    usacConfig->setRandomGeneratorState(state);
    usacConfig->setNumberOfThreads(number_of_threads);
    usacConfig->setNeighborsSearchMethod(_neighbors_search_method);
    usacConfig->setSamplingMethod(_sampling_method);
    usacConfig->setScoreMethod(_score_method);
    usacConfig->setLoMethod(_lo_method);
    // The mask is needed to remove the points of the model that has been segmented
    usacConfig->maskRequired(true);


    UniversalRANSAC ransac(usacConfig, points_size, estimator, quality, sampler,
            termination, verifier, degeneracy, lo, polisher);
    Ptr <usac::RansacOutput> ransac_output;

    if (!ransac.run(ransac_output))
    {
        return 0;
    }

    model_coefficients = ransac_output->getModel();
    label = ransac_output->getInliersMask();
    return ransac_output->getNumberOfInliers();
}

//-------------------------- segment -----------------------
int
SACSegmentation::segment(InputArray input_pts, OutputArray labels, OutputArray models_coefficients)
{
    Mat points;
    _getMatFromInputArray(input_pts, points);
    int pts_size = points.rows * points.cols / 3;

    std::vector<int> _labels(pts_size, 0);
    std::vector<Mat> _models_coefficients;


    // Keep the index array of the point corresponding to the original point
    AutoBuffer<int> ori_pts_idx(pts_size);
    int *pts_idx_ptr = ori_pts_idx.data();
    for (int i = 0; i < pts_size; ++i) pts_idx_ptr[i] = i;

    for (int model_num = 1; model_num <= number_of_models_expected; ++model_num)
    {
        Mat model_coefficients;
        std::vector<bool> label;

        int best_inls = segmentSingle(points, label, model_coefficients);
        if (best_inls < 1)
            break;

        _models_coefficients.emplace_back(model_coefficients);

        if (model_num != number_of_models_expected)
        {
            cv::Mat tmp_pts(points);
            points = cv::Mat(pts_size - best_inls, 3, CV_32F);

            float *const tmp_pts_ptr = (float *) tmp_pts.data;
            float *const pts_ptr = (float *) points.data;
            for (int j = 0, k = 0; k < pts_size; ++k)
            {
                if (label[k])
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
            pts_size = pts_size - best_inls;
        }
        else
        {
            for (int k = 0; k < pts_size; ++k)
            {
                if (label[k])
                    _labels[pts_idx_ptr[k]] = model_num;
            }
        }
    }

    int number_of_models = (int) _models_coefficients.size();
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


    //    Mat(_models_coefficients).copyTo(models_coefficients);
    if (models_coefficients.needed())
    {
        if (number_of_models != 0)
        {

            models_coefficients.create(number_of_models, 1, _models_coefficients[0].type());
            /**get vector*/
            std::vector<Mat> dst;
            models_coefficients.getMatVector(dst);
            for (int i = 0; i < number_of_models; i++)
            {
                Mat cur_mat = _models_coefficients[i];
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