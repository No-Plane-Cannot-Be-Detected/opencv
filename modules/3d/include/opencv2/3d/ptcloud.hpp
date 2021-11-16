// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.


#ifndef OPENCV_3D_PTCLOUD_HPP
#define OPENCV_3D_PTCLOUD_HPP

namespace cv {

//! @addtogroup _3d
//! @{

//! Custom function that take the model coefficients and return whether the model is acceptable or not
using ModelConstraintFunctionPtr = bool (*)(const Mat &/*model_coefficients*/);


//! type of the robust estimation algorithm
enum SacMethod
{
    /** "Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and
     * Automated Cartography", Martin A. Fischler and Robert C. Bolles, Comm. Of the ACM 24: 381–395, June 1981.
     */
    SAC_METHOD_RANSAC,
    //    SAC_METHOD_MAGSAC,
    //    SAC_METHOD_LMEDS,
    //    SAC_METHOD_MSAC,
    //    SAC_METHOD_RRANSAC,
    //    SAC_METHOD_RMSAC,
    //    SAC_METHOD_MLESAC,
    //    SAC_METHOD_PROSAC
};

enum SacModelType
{
    SAC_MODEL_PLANE,
    //    SAC_MODEL_SPHERE,
    //    SAC_MODEL_CYLINDER,

};

class CV_EXPORTS SACSegmentation : public Algorithm
{
public:
    using PointCloud = Mat;

    //! No-argument constructor using default configuration
    SACSegmentation()
            : sac_model_type(SAC_MODEL_PLANE), sac_method(SAC_METHOD_RANSAC), threshold(0),
              max_iterations(1000), probability(0.99), number_of_models_expected(1),
              number_of_threads(-1), rng_state(0),
              custom_model_constraints(nullptr)
    {
    }

    ~SACSegmentation() override = default;

    void setPointCloud(const Mat &input_pts);

    //-------------------------- Getter and Setter -----------------------

    /**
     * @brief Get the type of sample consensus model to use
     *
     * @param sac_model_type
     */
    inline void setSacModelType(SacModelType sac_model_type_)
    {
        sac_model_type = sac_model_type_;
    }

    //~ Get the type of sample consensus model used.
    inline SacModelType getSacModelType() const
    {
        return sac_model_type;
    }

    /**
     * @brief Set the type of sample consensus method to use.
     *
     * @param sac_method_
     */
    inline void setSacMethodType(SacMethod sac_method_)
    {
        sac_method = sac_method_;
    }

    //! Get the type of sample consensus method used.
    inline SacMethod getSacMethodType() const
    {
        return sac_method;
    }

    //! Set the distance to the model threshold.
    inline void setDistanceThreshold(double threshold_)
    {
        threshold = threshold_;
    }

    //! Get the distance to the model threshold.
    inline double getDistanceThreshold() const
    {
        return threshold;
    }

    //! Set the maximum number of iterations to attempt.
    inline void setMaxIterations(int max_iterations_)
    {
        max_iterations = max_iterations_;
    }

    //! Get the maximum number of iterations to attempt.
    inline int getMaxIterations() const
    {
        return max_iterations;
    }

    //! Set the probability that ensure at least one of selections is an error-free set of data points.
    inline void setProbability(double probability_)
    {
        probability = probability_;
    }

    //! Get the probability that ensure at least one of selections is an error-free set of data points.
    inline double getProbability() const
    {
        return probability;
    }

    //! Set the number of models expected.
    inline void setNumberOfModelsExpected(int number_of_models_expected_)
    {
        number_of_models_expected = number_of_models_expected_;
    }

    //! Get the expected number of models.
    inline int getNumberOfModelsExpected() const
    {
        return number_of_models_expected;
    }

    /**
     * @brief Set the number of threads to be used.
     *
     * @param number_of_threads_ The number of threads to be used.
     * (0 sets the value automatically, a negative number turns parallelization off)
     *
     * @note Not all SAC methods have a parallel implementation. Some will ignore this setting.
     */
    inline void setNumberOfThreads(int number_of_threads_)
    {
        number_of_threads = number_of_threads_;
    }

    // Get the number of threads to be used.
    inline int getNumberOfThreads() const
    {
        return number_of_threads;
    }

    //! Set state used to initialize the RNG(Random Number Generator).
    inline void setRandomGeneratorState(uint64 rng_state_)
    {
        rng_state = rng_state_;
    }

    //! Get state used to initialize the RNG(Random Number Generator).
    inline uint64 getRandomGeneratorState() const
    {
        return rng_state;
    }

    //! Set custom model coefficient constraint function
    inline void setCustomModelConstraints(ModelConstraintFunctionPtr custom_model_constraints_)
    {
        custom_model_constraints = custom_model_constraints_;
    }

    //! Get custom model coefficient constraint function
    inline ModelConstraintFunctionPtr getCustomModelConstraints() const
    {
        return custom_model_constraints;
    }

    /**
     * @brief Execute segmentation using the sample consensus method.
     *
     * @param[out] labels The label corresponds to the model number, 0 means it
     * does not belong to any model, range [0, Number of final resultant models obtained].
     * @param[out] models_coefficients The resultant models coefficients.
     * @return Number of final resultant models obtained by segmentation.
     */
    int segment(OutputArray labels, OutputArray models_coefficients);

protected:

    //! Point cloud data.
    PointCloud input_pts;

    //! The type of sample consensus model used.
    SacModelType sac_model_type;

    //! The type of sample consensus method used.
    SacMethod sac_method;

    //! Considered as inlier point if distance to the model less than threshold.
    double threshold;

    //!  The maximum number of iterations to attempt.
    int max_iterations;

    //! Probability that ensure at least one of selections is an error-free set of data points.
    double probability;

    //! Expected number of models.
    int number_of_models_expected;

    //! The number of threads the scheduler should use, or a negative number if no parallelization is wanted.
    int number_of_threads;

    //! 64-bit value used to initialize the RNG(Random Number Generator).
    uint64 rng_state;

    //! A user defined function that takes model coefficients and returns whether the model is acceptable or not.
    ModelConstraintFunctionPtr custom_model_constraints;

};


/**
 * @brief Point cloud sampling by Voxel Grid filter downsampling.
 *
 * Creates a 3D voxel grid (a set of tiny 3D boxes in space) over the input
 * point cloud data, in each voxel (i.e., 3D box), all the points present will be
 * approximated (i.e., downsampled) with the point closest to their centroid.
 *
 * @param sampled_point_flags  (Output) Flags of the sampled point, (pass in std::vector<int> or std::vector<char> etc.)
 *                     sampled_point_flags[i] is 1 means i-th point selected, 0 means it is not selected.
 * @param input_pts  Original point cloud, vector of Point3 or Mat of size Nx3/3xN.
 * @param length Grid length.
 * @param width  Grid width.
 * @param height  Grid height.
 * @return The number of points actually sampled.
 */
CV_EXPORTS int voxelGridSampling(OutputArray sampled_point_flags, InputArray input_pts,
        float length, float width, float height);

/**
 * @brief Point cloud sampling by randomly select points.
 *
 * Use cv::randShuffle to shuffle the point index list,
 * then take the points corresponding to the front part of the list.
 *
 * @param sampled_pts  Point cloud after sampling.
 *                     Support cv::Mat(sampled_pts_size, 3, CV_32F), std::vector<cv::Point3f>.
 * @param input_pts  Original point cloud, vector of Point3 or Mat of size Nx3/3xN.
 * @param sampled_pts_size The desired point cloud size after sampling.
 * @param rng  Optional random number generator used for cv::randShuffle;
 *                      if it is nullptr, theRNG () is used instead.
 */
CV_EXPORTS void randomSampling(OutputArray sampled_pts, InputArray input_pts,
        int sampled_pts_size, RNG *rng = nullptr);

/**
 * @overload
 *
 * @param sampled_pts  Point cloud after sampling.
 *                     Support cv::Mat(size * sampled_scale, 3, CV_32F), std::vector<cv::Point3f>.
 * @param input_pts  Original point cloud, vector of Point3 or Mat of size Nx3/3xN.
 * @param sampled_scale Range (0, 1), the percentage of the sampled point cloud to the original size,
 *                      that is, sampled size = original size * sampled_scale.
 * @param rng  Optional random number generator used for cv::randShuffle;
 *                      if it is nullptr, theRNG () is used instead.
 */
CV_EXPORTS void randomSampling(OutputArray sampled_pts, InputArray input_pts,
        float sampled_scale, RNG *rng = nullptr);

/**
 * @brief Point cloud sampling by Farthest Point Sampling(FPS).
 *
 * FPS Algorithm:
 *   Input: Point cloud *C*, *sampled_pts_size*, *dist_lower_limit*
 *   Initialize: Set sampled point cloud S to the empty set
 *   Step:
 *     1. Randomly take a seed point from C and take it from C to S;
 *     2. Find a point in C that is the farthest away from S and take it from C to S;
 *       (The distance from point to set S is the smallest distance from point to all points in S)
 *     3. Repeat *step 2* until the farthest distance of the point in C from S
 *       is less than *dist_lower_limit*, or the size of S is equal to *sampled_pts_size*.
 *   Output: Sampled point cloud S
 *
 * @param sampled_point_flags  (Output) Flags of the sampled point, (pass in std::vector<int> or std::vector<char> etc.)
 *                     sampled_point_flags[i] is 1 means i-th point selected, 0 means it is not selected.
 * @param input_pts  Original point cloud, vector of Point3 or Mat of size Nx3/3xN.
 * @param sampled_pts_size The desired point cloud size after sampling.
 * @param dist_lower_limit Sampling is terminated early if the distance from
 *                  the farthest point to S is less than dist_lower_limit, default 0.
 * @param rng Optional random number generator used for selecting seed point for FPS;
 *                  if it is nullptr, theRNG () is used instead.
 * @return The number of points actually sampled.
 */
CV_EXPORTS int farthestPointSampling(OutputArray sampled_point_flags, InputArray input_pts,
        int sampled_pts_size, float dist_lower_limit = 0, RNG *rng = nullptr);

/**
 * @overload
 *
 * @param sampled_point_flags  (Output) Flags of the sampled point, (pass in std::vector<int> or std::vector<char> etc.)
 *                     sampled_point_flags[i] is 1 means i-th point selected, 0 means it is not selected.
 * @param input_pts  Original point cloud, vector of Point3 or Mat of size Nx3/3xN.
 * @param sampled_scale Range (0, 1), the percentage of the sampled point cloud to the original size,
 *                      that is, sampled size = original size * sampled_scale.
 * @param dist_lower_limit Sampling is terminated early if the distance from
 *                  the farthest point to S is less than dist_lower_limit, default 0.
 * @param rng Optional random number generator used for selecting seed point for FPS;
 *                  if it is nullptr, theRNG () is used instead.
 * @return The number of points actually sampled.
 */
CV_EXPORTS int farthestPointSampling(OutputArray sampled_point_flags, InputArray input_pts,
        float sampled_scale, float dist_lower_limit = 0, RNG *rng = nullptr);

//! @} _3d
} //end namespace cv
#endif //OPENCV_3D_PTCLOUD_HPP
