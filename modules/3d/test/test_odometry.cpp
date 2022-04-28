// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "test_precomp.hpp"

namespace opencv_test { namespace {

static
void warpFrame(const Mat& image, const Mat& depth, const Mat& rvec, const Mat& tvec, const Mat& K,
               Mat& warpedImage, Mat& warpedDepth)
{
    CV_Assert(!image.empty());
    CV_Assert(image.type() == CV_8UC1);

    CV_Assert(depth.size() == image.size());
    CV_Assert(depth.type() == CV_32FC1);

    CV_Assert(!rvec.empty());
    CV_Assert(rvec.total() == 3);
    CV_Assert(rvec.type() == CV_64FC1);

    CV_Assert(!tvec.empty());
    CV_Assert(tvec.size() == Size(1, 3));
    CV_Assert(tvec.type() == CV_64FC1);

    warpedImage.create(image.size(), CV_8UC1);
    warpedImage = Scalar(0);
    warpedDepth.create(image.size(), CV_32FC1);
    warpedDepth = Scalar(FLT_MAX);

    Mat cloud;
    depthTo3d(depth, K, cloud);

    Mat cloud3, channels[4];
    cv::split(cloud, channels);
    std::vector<Mat> merged = { channels[0], channels[1], channels[2] };
    cv::merge(merged, cloud3);

    Mat Rt = Mat::eye(4, 4, CV_64FC1);
    {
        Mat R, dst;
        cv::Rodrigues(rvec, R);

        dst = Rt(Rect(0,0,3,3));
        R.copyTo(dst);

        dst = Rt(Rect(3,0,1,3));
        tvec.copyTo(dst);
    }
    Mat warpedCloud, warpedImagePoints;
    perspectiveTransform(cloud3, warpedCloud, Rt);
    projectPoints(warpedCloud.reshape(3, 1), Mat(3,1,CV_32FC1, Scalar(0)), Mat(3,1,CV_32FC1, Scalar(0)), K, Mat(1,5,CV_32FC1, Scalar(0)), warpedImagePoints);
    warpedImagePoints = warpedImagePoints.reshape(2, cloud.rows);
    Rect r(0, 0, image.cols, image.rows);
    for(int y = 0; y < cloud.rows; y++)
    {
        for(int x = 0; x < cloud.cols; x++)
        {
            Point p = warpedImagePoints.at<Point2f>(y,x);
            if(r.contains(p))
            {
                float curDepth = warpedDepth.at<float>(p.y, p.x);
                float newDepth = warpedCloud.at<Point3f>(y, x).z;
                if(newDepth < curDepth && newDepth > 0)
                {
                    warpedImage.at<uchar>(p.y, p.x) = image.at<uchar>(y,x);
                    warpedDepth.at<float>(p.y, p.x) = newDepth;
                }
            }
        }
    }
    warpedDepth.setTo(std::numeric_limits<float>::quiet_NaN(), warpedDepth > 100);
}

static
void dilateFrame(Mat& image, Mat& depth)
{
    CV_Assert(!image.empty());
    CV_Assert(image.type() == CV_8UC1);

    CV_Assert(!depth.empty());
    CV_Assert(depth.type() == CV_32FC1);
    CV_Assert(depth.size() == image.size());

    Mat mask(image.size(), CV_8UC1, Scalar(255));
    for(int y = 0; y < depth.rows; y++)
        for(int x = 0; x < depth.cols; x++)
            if(cvIsNaN(depth.at<float>(y,x)) || depth.at<float>(y,x) > 10 || depth.at<float>(y,x) <= FLT_EPSILON)
                mask.at<uchar>(y,x) = 0;

    image.setTo(255, ~mask);
    Mat minImage;
    erode(image, minImage, Mat());

    image.setTo(0, ~mask);
    Mat maxImage;
    dilate(image, maxImage, Mat());

    depth.setTo(FLT_MAX, ~mask);
    Mat minDepth;
    erode(depth, minDepth, Mat());

    depth.setTo(0, ~mask);
    Mat maxDepth;
    dilate(depth, maxDepth, Mat());

    Mat dilatedMask;
    dilate(mask, dilatedMask, Mat(), Point(-1,-1), 1);
    for(int y = 0; y < depth.rows; y++)
        for(int x = 0; x < depth.cols; x++)
            if(!mask.at<uchar>(y,x) && dilatedMask.at<uchar>(y,x))
            {
                image.at<uchar>(y,x) = static_cast<uchar>(0.5f * (static_cast<float>(minImage.at<uchar>(y,x)) +
                                                                  static_cast<float>(maxImage.at<uchar>(y,x))));
                depth.at<float>(y,x) = 0.5f * (minDepth.at<float>(y,x) + maxDepth.at<float>(y,x));
            }
}

class OdometryTest
{
public:
    OdometryTest(OdometryType _otype,
                 OdometryAlgoType _algtype,
                 double _maxError1,
                 double _maxError5,
                 double _idError = DBL_EPSILON) :
        otype(_otype),
        algtype(_algtype),
        maxError1(_maxError1),
        maxError5(_maxError5),
        idError(_idError)
    { }

    void readData(Mat& image, Mat& depth) const;
    static Mat getCameraMatrix()
    {
        float fx = 525.0f, // default
              fy = 525.0f,
              cx = 319.5f,
              cy = 239.5f;
        Matx33f K(fx,  0, cx,
                   0, fy, cy,
                   0,  0,  1);
        return Mat(K);
    }
    static void generateRandomTransformation(Mat& R, Mat& t);

    void run();
    void checkUMats();
    void prepareFrameCheck();

    OdometryType otype;
    OdometryAlgoType algtype;
    double maxError1;
    double maxError5;
    double idError;
};


void OdometryTest::readData(Mat& image, Mat& depth) const
{
    std::string dataPath = cvtest::TS::ptr()->get_data_path();
    std::string imageFilename = dataPath + "/cv/rgbd/rgb.png";
    std::string depthFilename = dataPath + "/cv/rgbd/depth.png";

    image = imread(imageFilename,  0);
    depth = imread(depthFilename, -1);

    if(image.empty())
    {
        FAIL() << "Image " << imageFilename.c_str() << " can not be read" << std::endl;
    }
    if(depth.empty())
    {
        FAIL() << "Depth" << depthFilename.c_str() << "can not be read" << std::endl;
    }

    CV_DbgAssert(image.type() == CV_8UC1);
    CV_DbgAssert(depth.type() == CV_16UC1);
    {
        Mat depth_flt;
        depth.convertTo(depth_flt, CV_32FC1, 1.f/5000.f);
        depth_flt.setTo(std::numeric_limits<float>::quiet_NaN(), depth_flt < FLT_EPSILON);
        depth = depth_flt;
    }
}

void OdometryTest::generateRandomTransformation(Mat& rvec, Mat& tvec)
{
    const float maxRotation = (float)(3.f / 180.f * CV_PI); //rad
    const float maxTranslation = 0.02f; //m

    RNG& rng = theRNG();
    rvec.create(3, 1, CV_64FC1);
    tvec.create(3, 1, CV_64FC1);

    randu(rvec, Scalar(-1000), Scalar(1000));
    normalize(rvec, rvec, rng.uniform(0.007f, maxRotation));

    randu(tvec, Scalar(-1000), Scalar(1000));
    normalize(tvec, tvec, rng.uniform(0.008f, maxTranslation));
}

void OdometryTest::checkUMats()
{
    Mat K = getCameraMatrix();

    Mat image, depth;
    readData(image, depth);

    OdometrySettings ods;
    ods.setCameraMatrix(K);
    Odometry odometry = Odometry(otype, ods, algtype);
    OdometryFrame odf = odometry.createOdometryFrame(OdometryFrameStoreType::UMAT);

    Mat calcRt;

    UMat uimage, udepth;
    image.copyTo(uimage);
    depth.copyTo(udepth);
    odf.setImage(uimage);
    odf.setDepth(udepth);
    uimage.release();
    udepth.release();

    odometry.prepareFrame(odf);
    bool isComputed = odometry.compute(odf, odf, calcRt);
    ASSERT_TRUE(isComputed);
    double diff = cv::norm(calcRt, Mat::eye(4, 4, CV_64FC1));
    if (diff > idError)
    {
        FAIL() << "Incorrect transformation between the same frame (not the identity matrix), diff = " << diff << std::endl;
    }

}

void OdometryTest::run()
{
    Mat K = getCameraMatrix();

    Mat image, depth;
    readData(image, depth);
    OdometrySettings ods;
    ods.setCameraMatrix(K);
    Odometry odometry = Odometry(otype, ods, algtype);
    OdometryFrame odf = odometry.createOdometryFrame();
    odf.setImage(image);
    odf.setDepth(depth);
    Mat calcRt;

    // 1. Try to find Rt between the same frame (try masks also).
    Mat mask(image.size(), CV_8UC1, Scalar(255));

    odometry.prepareFrame(odf);
    bool isComputed = odometry.compute(odf, odf, calcRt);

    if(!isComputed)
    {
        FAIL() << "Can not find Rt between the same frame" << std::endl;
    }
    double ndiff = cv::norm(calcRt, Mat::eye(4,4,CV_64FC1));
    if(ndiff > idError)
    {
        FAIL() << "Incorrect transformation between the same frame (not the identity matrix), diff = " << ndiff << std::endl;
    }

    // 2. Generate random rigid body motion in some ranges several times (iterCount).
    // On each iteration an input frame is warped using generated transformation.
    // Odometry is run on the following pair: the original frame and the warped one.
    // Comparing a computed transformation with an applied one we compute 2 errors:
    // better_1time_count - count of poses which error is less than ground truth pose,
    // better_5times_count - count of poses which error is 5 times less than ground truth pose.
    int iterCount = 100;
    int better_1time_count = 0;
    int better_5times_count = 0;
    for (int iter = 0; iter < iterCount; iter++)
    {
        Mat rvec, tvec;
        generateRandomTransformation(rvec, tvec);

        Mat warpedImage, warpedDepth;
        warpFrame(image, depth, rvec, tvec, K, warpedImage, warpedDepth);
        dilateFrame(warpedImage, warpedDepth); // due to inaccuracy after warping

        OdometryFrame odfSrc = odometry.createOdometryFrame();
        OdometryFrame odfDst = odometry.createOdometryFrame();
        odfSrc.setImage(image);
        odfSrc.setDepth(depth);
        odfDst.setImage(warpedImage);
        odfDst.setDepth(warpedDepth);

        odometry.prepareFrames(odfSrc, odfDst);
        isComputed = odometry.compute(odfSrc, odfDst, calcRt);

        if (!isComputed)
            continue;
        Mat calcR = calcRt(Rect(0,0,3,3)), calcRvec;
        cv::Rodrigues(calcR, calcRvec);
        calcRvec = calcRvec.reshape(rvec.channels(), rvec.rows);
        Mat calcTvec = calcRt(Rect(3,0,1,3));

        if (cvtest::debugLevel >= 10)
        {
            imshow("image", image);
            imshow("warpedImage", warpedImage);
            Mat resultImage, resultDepth;
            warpFrame(image, depth, calcRvec, calcTvec, K, resultImage, resultDepth);
            imshow("resultImage", resultImage);
            waitKey(100);
        }

        // compare rotation
        double possibleError = algtype == OdometryAlgoType::COMMON ? 0.11f : 0.015f;

        Affine3f src = Affine3f(Vec3f(rvec), Vec3f(tvec));
        Affine3f res = Affine3f(Vec3f(calcRvec), Vec3f(calcTvec));
        Affine3f src_inv = src.inv();
        Affine3f diff = res * src_inv;
        double rdiffnorm = cv::norm(diff.rvec());
        double tdiffnorm = cv::norm(diff.translation());

        if (rdiffnorm < possibleError && tdiffnorm < possibleError)
        {
            better_1time_count++;
        }
        if (5. * rdiffnorm < possibleError && 5 * tdiffnorm < possibleError)
            better_5times_count++;

        CV_LOG_INFO(NULL, "Iter " << iter);
        CV_LOG_INFO(NULL, "rdiff: " << Vec3f(diff.rvec()) << "; rdiffnorm: " << rdiffnorm);
        CV_LOG_INFO(NULL, "tdiff: " << Vec3f(diff.translation()) << "; tdiffnorm: " << tdiffnorm);

        CV_LOG_INFO(NULL, "better_1time_count " << better_1time_count << "; better_5time_count " << better_5times_count);
    }

    if(static_cast<double>(better_1time_count) < maxError1 * static_cast<double>(iterCount))
    {
        FAIL() << "Incorrect count of accurate poses [1st case]: "
            << static_cast<double>(better_1time_count) << " / "
            << maxError1 * static_cast<double>(iterCount) << std::endl;
    }

    if(static_cast<double>(better_5times_count) < maxError5 * static_cast<double>(iterCount))
    {
        FAIL() << "Incorrect count of accurate poses [2nd case]: "
            << static_cast<double>(better_5times_count) << " / "
            << maxError5 * static_cast<double>(iterCount) << std::endl;
    }
}

void OdometryTest::prepareFrameCheck()
{
    Mat K = getCameraMatrix();

    Mat image, depth;
    readData(image, depth);
    OdometrySettings ods;
    ods.setCameraMatrix(K);
    Odometry odometry = Odometry(otype, ods, algtype);
    OdometryFrame odf = odometry.createOdometryFrame();
    odf.setImage(image);
    odf.setDepth(depth);

    odometry.prepareFrame(odf);

    Mat points, mask;
    odf.getPyramidAt(points, OdometryFramePyramidType::PYR_CLOUD, 0);
    odf.getPyramidAt(mask, OdometryFramePyramidType::PYR_MASK, 0);

    OdometryFrame todf = odometry.createOdometryFrame();
    if (otype != OdometryType::DEPTH)
    {
        Mat img;
        odf.getPyramidAt(img, OdometryFramePyramidType::PYR_IMAGE, 0);
        todf.setPyramidLevel(1, OdometryFramePyramidType::PYR_IMAGE);
        todf.setPyramidAt(img, OdometryFramePyramidType::PYR_IMAGE, 0);
    }
    todf.setPyramidLevel(1, OdometryFramePyramidType::PYR_CLOUD);
    todf.setPyramidAt(points, OdometryFramePyramidType::PYR_CLOUD, 0);
    todf.setPyramidLevel(1, OdometryFramePyramidType::PYR_MASK);
    todf.setPyramidAt(mask, OdometryFramePyramidType::PYR_MASK, 0);

    odometry.prepareFrame(todf);
}

/****************************************************************************************\
*                                Tests registrations                                     *
\****************************************************************************************/

TEST(RGBD_Odometry_Rgbd, algorithmic)
{
    OdometryTest test(OdometryType::RGB, OdometryAlgoType::COMMON, 0.99, 0.89);
    test.run();
}

TEST(RGBD_Odometry_ICP, algorithmic)
{
    OdometryTest test(OdometryType::DEPTH, OdometryAlgoType::COMMON, 0.99, 0.99);
    test.run();
}

TEST(RGBD_Odometry_RgbdICP, algorithmic)
{
    OdometryTest test(OdometryType::RGB_DEPTH, OdometryAlgoType::COMMON, 0.99, 0.99);
    test.run();
}

TEST(RGBD_Odometry_FastICP, algorithmic)
{
    OdometryTest test(OdometryType::DEPTH, OdometryAlgoType::FAST, 0.99, 0.89, FLT_EPSILON);
    test.run();
}


TEST(RGBD_Odometry_Rgbd, UMats)
{
    OdometryTest test(OdometryType::RGB, OdometryAlgoType::COMMON, 0.99, 0.89);
    test.checkUMats();
}

TEST(RGBD_Odometry_ICP, UMats)
{
    OdometryTest test(OdometryType::DEPTH, OdometryAlgoType::COMMON, 0.99, 0.99);
    test.checkUMats();
}

TEST(RGBD_Odometry_RgbdICP, UMats)
{
    OdometryTest test(OdometryType::RGB_DEPTH, OdometryAlgoType::COMMON, 0.99, 0.99);
    test.checkUMats();
}

TEST(RGBD_Odometry_FastICP, UMats)
{
    OdometryTest test(OdometryType::DEPTH, OdometryAlgoType::FAST, 0.99, 0.89, FLT_EPSILON);
    test.checkUMats();
}


TEST(RGBD_Odometry_Rgbd, prepareFrame)
{
    OdometryTest test(OdometryType::RGB, OdometryAlgoType::COMMON, 0.99, 0.89);
    test.prepareFrameCheck();
}

TEST(RGBD_Odometry_ICP, prepareFrame)
{
    OdometryTest test(OdometryType::DEPTH, OdometryAlgoType::COMMON, 0.99, 0.99);
    test.prepareFrameCheck();
}

TEST(RGBD_Odometry_RgbdICP, prepareFrame)
{
    OdometryTest test(OdometryType::RGB_DEPTH, OdometryAlgoType::COMMON, 0.99, 0.99);
    test.prepareFrameCheck();
}

TEST(RGBD_Odometry_FastICP, prepareFrame)
{
    OdometryTest test(OdometryType::DEPTH, OdometryAlgoType::FAST, 0.99, 0.89, FLT_EPSILON);
    test.prepareFrameCheck();
}

}} // namespace
