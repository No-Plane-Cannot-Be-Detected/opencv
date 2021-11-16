// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.


#ifndef OPENCV_3D_PTCLOUD_UTILS_HPP
#define OPENCV_3D_PTCLOUD_UTILS_HPP

namespace cv {

/**
 * Get cv::Mat with type N×3 CV_32FC1 from cv::InputArray
 * Use different interpretations for the same memory data.
 */
inline void _getMatFromInputArray(InputArray input_pts, Mat &mat)
{
    CV_Check(input_pts.dims(), input_pts.dims() < 3,
             "Only support data with dimension less than 3.");

    // Guaranteed data can construct N×3 point clouds
    int rows = input_pts.rows(), cols = input_pts.cols(), channels = input_pts.channels();
    size_t total = rows * cols * channels;
    CV_Check(total, total % 3 == 0,
             "total = input_pts.rows() * input_pts.cols() * input_pts.channels() must be an integer multiple of 3");

    if (channels == 1 && rows == 3 && cols != 3)
    {
        // Layout of point cloud data in memory space:
        // x1, ..., xn, y1, ..., yn, z1, ..., zn
        // For example, the input is cv::Mat with type 3×N CV_32FC1
        transpose(input_pts, mat);
    }
    else
    {
        // Layout of point cloud data in memory space:
        // x1, y1, z1, ..., xn, yn, zn
        // For example, the input is std::vector<Point3d>, or std::vector<int>, or cv::Mat with type N×1 CV_32FC3
        mat = input_pts.getMat().reshape(1, (int) (total / 3));
    }

    if (mat.type() != CV_32F)
    {
        Mat tmp;
        mat.convertTo(tmp, CV_32F); // Use float to store data
        swap(mat, tmp);
    }

    if (!mat.isContinuous())
    {
        mat = mat.clone();
    }

}

inline void copyPointDataByIdxs(const Mat &src, Mat &dst, const std::vector<int> &idxs, int dst_size = -1)
{
    if (-1 == dst_size)
    {
        dst_size = (int) idxs.size();
    }
    dst = Mat(dst_size, 3, CV_32F);

    const float *src_ptr = (float *) src.data;
    float *const dst_ptr = (float *) dst.data;

    for (int i = 0; i < dst_size; ++i)
    {
        const float *src_ptr_base = src_ptr + 3 * idxs[i];
        float *dst_ptr_base = dst_ptr + 3 * i;
        dst_ptr_base[0] = src_ptr_base[0];
        dst_ptr_base[1] = src_ptr_base[1];
        dst_ptr_base[2] = src_ptr_base[2];
    }
}

inline void copyPointDataByFlags(const Mat &src, Mat &dst, const std::vector<bool> &flags)
{
    int pt_size = (int) flags.size();
    std::vector<int> idxs;
    for (int i = 0; i < pt_size; ++i)
    {
        if (flags[i])
        {
            idxs.emplace_back(i);
        }
    }
    copyPointDataByIdxs(src, dst, idxs, -1);
}

}

#endif //OPENCV_3D_PTCLOUD_UTILS_HPP
