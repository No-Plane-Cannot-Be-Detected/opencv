// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../usac.hpp"

namespace cv { namespace usac {

class SphereModelMinimalSolverImpl : public SphereModelMinimalSolver
{
private:
    const Mat *points_mat;
    const float *const points;

public:
    explicit SphereModelMinimalSolverImpl(const Mat &points_)
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
        return makePtr<SphereModelMinimalSolverImpl>(*points_mat);
    }

    int estimate(const std::vector<int> &sample, std::vector<Mat> &models) const override
    {
        // TODO
        return 0;
    }
};


Ptr <SphereModelMinimalSolver> SphereModelMinimalSolver::create(const Mat &points_)
{
    return makePtr<SphereModelMinimalSolverImpl>(points_);
}


class SphereModelNonMinimalSolverImpl : public SphereModelNonMinimalSolver
{
private:
    const Mat *points_mat;
    const float *const points;

public:
    explicit SphereModelNonMinimalSolverImpl(const Mat &points_)
            : points_mat(&points_), points((float *) points_.data)
    {
    }

    int getMinimumRequiredSampleSize() const override
    {
        return 3;
    }

    int getMaxNumberOfSolutions() const override
    {
        // TODO
        return 0;
    }

    Ptr <NonMinimalSolver> clone() const override
    {
        return makePtr<SphereModelNonMinimalSolverImpl>(*points_mat);
    }

    int estimate(const std::vector<int> &sample, int sample_size, std::vector<Mat> &models,
            const std::vector<double> &weights) const override
    {
        // TODO
        return 0;
    }
};

Ptr <SphereModelNonMinimalSolver> SphereModelNonMinimalSolver::create(const Mat &points_)
{
    return makePtr<SphereModelNonMinimalSolverImpl>(points_);
}

}}