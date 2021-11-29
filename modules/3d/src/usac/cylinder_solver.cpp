// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../ptcloud/ptcloud_wrapper.hpp"
#include "../usac.hpp"

namespace cv { namespace usac {

class CylinderModelMinimalSolverImpl : public CylinderModelMinimalSolver, public PointCloudWrapper
{

public:
    explicit CylinderModelMinimalSolverImpl(const Mat &points_)
            : PointCloudWrapper(points_)
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
        return makePtr<CylinderModelMinimalSolverImpl>(*points_mat);
    }

    int estimate(const std::vector<int> &sample, std::vector<Mat> &models) const override
    {
        // TODO
        return 0;
    }
};


Ptr <CylinderModelMinimalSolver> CylinderModelMinimalSolver::create(const Mat &points_)
{
    return makePtr<CylinderModelMinimalSolverImpl>(points_);
}


class CylinderModelNonMinimalSolverImpl : public CylinderModelNonMinimalSolver, public PointCloudWrapper
{

public:
    explicit CylinderModelNonMinimalSolverImpl(const Mat &points_)
            : PointCloudWrapper(points_)
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
        return makePtr<CylinderModelNonMinimalSolverImpl>(*points_mat);
    }

    int estimate(const std::vector<int> &sample, int sample_size, std::vector<Mat> &models,
            const std::vector<double> &weights) const override
    {
        // TODO
        return 0;
    }
};

Ptr <CylinderModelNonMinimalSolver> CylinderModelNonMinimalSolver::create(const Mat &points_)
{
    return makePtr<CylinderModelNonMinimalSolverImpl>(points_);
}

}}