// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../usac.hpp"
#include "../ptcloud/ptcloud_utils.hpp"

namespace cv { namespace usac {

class PlaneModelMinimalSolverImpl : public PlaneModelMinimalSolver
{
private:
    const Mat *points_mat;
    const float *const points;

public:
    explicit PlaneModelMinimalSolverImpl(const Mat &points_)
            : points_mat(&points_), points((float *) points_.data)
    {
    }

    int getSampleSize() const override
    {
        return 3;
    }

    int getMaxNumberOfSolutions() const override
    {
        return 1;
    }

    Ptr <MinimalSolver> clone() const override
    {
        return makePtr<PlaneModelMinimalSolverImpl>(*points_mat);
    }

    int estimate(const std::vector<int> &sample, std::vector<Mat> &models) const override
    {
        models.clear();

        // Get point data
        const float *p1_ptr_base = points + 3 * sample[0];
        const float *p2_ptr_base = points + 3 * sample[1];
        const float *p3_ptr_base = points + 3 * sample[2];
        float x1 = p1_ptr_base[0], y1 = p1_ptr_base[1], z1 = p1_ptr_base[2];
        float x2 = p2_ptr_base[0], y2 = p2_ptr_base[1], z2 = p2_ptr_base[2];
        float x3 = p3_ptr_base[0], y3 = p3_ptr_base[1], z3 = p3_ptr_base[2];

        // v1 = p1p2  v2 = p1p3
        float a1 = x2 - x1;
        float b1 = y2 - y1;
        float c1 = z2 - z1;
        float a2 = x3 - x1;
        float b2 = y3 - y1;
        float c2 = z3 - z1;

        // Get the plane normal vector v = v1 x v2
        float a = b1 * c2 - b2 * c1;
        float b = a2 * c1 - a1 * c2;
        float c = a1 * b2 - b1 * a2;
        float d = (-a * x1 - b * y1 - c * z1);

        double plane_coeff[4] = {a, b, c, d};
        models.emplace_back(cv::Mat(1, 4, CV_64F, plane_coeff).clone());

//        models = std::vector<Mat>{Mat_<double>(3, 3)};
//        auto *f = (double *) models[0].data;
//        f[0] = a, f[1] = b, f[2] = c, f[3] = d;

        return 1;
    }
};


Ptr <PlaneModelMinimalSolver> PlaneModelMinimalSolver::create(const Mat &points_)
{
    return makePtr<PlaneModelMinimalSolverImpl>(points_);
}


class PlaneModelNonMinimalSolverImpl : public PlaneModelNonMinimalSolver
{
private:
    const Mat *points_mat;
    const float *const points;

public:
    explicit PlaneModelNonMinimalSolverImpl(const Mat &points_)
            : points_mat(&points_), points((float *) points_.data)
    {
    }

    int getMinimumRequiredSampleSize() const override
    {
        return 3;
    }

    int getMaxNumberOfSolutions() const override
    {
        return 1;
    }

    Ptr <NonMinimalSolver> clone() const override
    {
        return makePtr<PlaneModelNonMinimalSolverImpl>(*points_mat);
    }

    int estimate(const std::vector<int> &sample, int sample_size, std::vector<Mat> &models,
            const std::vector<double> &weights) const override
    {
        models.clear();

        cv::Mat pcaset;

        copyPointDataByIdxs(*points_mat, pcaset, sample, sample_size);

        cv::PCA pca(pcaset, // pass the data
                    cv::Mat(), // we do not have a pre-computed mean vector,
                // so let the PCA engine to compute it
                    cv::PCA::DATA_AS_ROW, // indicate that the vectors
                // are stored as matrix rows
                // (use PCA::DATA_AS_COL if the vectors are
                // the matrix columns)
                    3 // specify, how many principal components to retain
        );


        Mat eigenvectors = pca.eigenvectors;
        const float *eig_ptr = (float *) eigenvectors.data;

        float a = eig_ptr[6], b = eig_ptr[7], c = eig_ptr[8];
        if (std::isinf(a) || std::isinf(b) || std::isinf(c) || (a == 0 && b == 0 && c == 0))
            return 0;

        Mat mean = pca.mean;
        const float *mean_ptr = (float *) mean.data;
        float d = (-a * mean_ptr[0] - b * mean_ptr[1] - c * mean_ptr[2]);

        double plane_coeffs[4] = {a, b, c, d};
        models.emplace_back(cv::Mat(1, 4, CV_64F, plane_coeffs).clone());
        return 1;
    }
};

Ptr <PlaneModelNonMinimalSolver> PlaneModelNonMinimalSolver::create(const Mat &points_)
{
    return makePtr<PlaneModelNonMinimalSolverImpl>(points_);
}

}}