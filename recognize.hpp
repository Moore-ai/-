#pragma once
#define RECOGNIZE_HPP

#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <vector>
#include <fstream>

using String = std::string;
using Mat = cv::Mat;
using Rect = cv::Rect;
using BGR = std::tuple<double,double,double>;

BGR make_BGR(const double b,const double g,const double r);

class Recognize {
    public:

    void test_A();

    void test_B();

    void test_C(const Mat&);
    void test_C_(int);

    void test_D();

    void test_E_();

    void go_video3();

    void loadModel();

    void loadParam(int);

    int make_predict(const Mat& img);

    BGR count_BGR_average(const Mat& image);

    double count_gray_average(const Mat& image);

    BGR special_BGR_average(const Mat& image,const Mat& mask);

    bool isYellow(const cv::Scalar& color);

    bool isYellow(const BGR& color);

    bool isColor(const BGR& color);

    cv::Point2d get_middle(const cv::Rect& rect);

    double count_distance(const cv::Point2d& a,const cv::Point2d& b);

    bool can_tolerate_area(const cv::Rect& a,const cv::Rect& b);

    bool can_tolerate_rate(const double a,const double b);
    bool can_tolerate_rate_hero(const double a,const double b);

    bool can_tolerate_height(const cv::Rect& a,const cv::Rect& b);

    bool can_tolerate_dist(const double,const double);
    bool can_tolerate_dist_hero(const double,const double);

    bool height_greater_than_width(const Rect& rect);

    double angle(const cv::Point2d&,const cv::Point2d&);
    double calculateAngle(const cv::Point2d& pointA, const cv::Point2d& pointB, const cv::Point2d& pointC);

    bool can_tolerate_angle(const double,const double);

    // RectInfo calculateRectInfo(const cv::Rect rect);

    // int can_tolerate(const RectInfo& rectInfo_A,const RectInfo& rectInfo_B);

    private:
    cv::VideoCapture video;
    std::vector<Mat>train_images;
    std::vector<int>labels;

    cv::Ptr<cv::ml::SVM>loadedSvm;

    std::vector<double>params;
};
