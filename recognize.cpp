#include "recognize.hpp"

static void getColor_gray(int event,int x,int y,int flags,void* frame);
static void getColor_BGR(int event,int x,int y,int flags,void* frame);

BGR make_BGR(const double b,const double g,const double r) {
    return std::make_tuple(b,g,r);
}

static std::vector<float> extractFeatures(const cv::Mat& img) {
    if (img.channels()!= 1) {
        std::cerr << "输入图像不是单通道灰度图像，不符合特征提取要求" << std::endl;
        throw std::runtime_error("图像通道数异常");
    }

    std::vector<float> features(img.total());  // 根据图像元素总数初始化features向量大小
    cv::Mat tempMat = cv::Mat(1, img.total(), CV_32F, features.data());  // 创建明确尺寸和类型的临时矩阵
    auto temp = img.reshape(0, 1);
    temp.convertTo(tempMat, CV_32F);  // 进行类型转换，确保数据类型匹配后再拷贝
    return features;
}

void Recognize::test_A() {
    Mat frame;
    while (true) {
        this->video>>frame;

        if (frame.empty()) break;

        Mat grayImg,gaussianedImg,cuttedImg;
        cv::cvtColor(frame,grayImg,cv::COLOR_BGR2GRAY);

        // cv::equalizeHist(grayImg,grayImg);

        cv::GaussianBlur(grayImg,gaussianedImg,cv::Size(3,3),0);
        cv::threshold(gaussianedImg,cuttedImg,127,255,cv::THRESH_BINARY);

        std::vector<std::vector<cv::Point>>contours;
        findContours(cuttedImg,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_NONE);
        cv::Mat emptyCanvas(grayImg.rows,grayImg.cols,CV_8UC1,cv::Scalar(255));
        // drawContours(emptyCanvas,contours,-1,cv::Scalar(0),2);

        std::vector<std::pair<cv::Rect,float>>rects;

        for (auto& contour:contours) {
            cv::Rect rect=cv::boundingRect(contour);
            if ((float)rect.height>=(float)rect.width*1.2) {
                Mat roi=grayImg(rect);
                float meanBrightness = cv::mean(roi)[0];

                cv::rectangle(frame,rect,cv::Scalar(0,255,0),2);
                cv::rectangle(grayImg,rect,cv::Scalar(0,255,0),2);
                cv::rectangle(cuttedImg,rect,cv::Scalar(0,255,0),2);
                cv::rectangle(emptyCanvas,rect,cv::Scalar(0,255,0),2);

                rects.push_back(std::make_pair(rect,meanBrightness));
            }
        }

        cv::imshow("frame",frame);
        cv::imshow("video",cuttedImg);
        cv::imshow("grayImg",grayImg);
        cv::imshow("canvas",emptyCanvas);

        cv::waitKey(25);
    }
    
}

void Recognize::test_B() {
    Mat img=cv::imread("/home/lenovo/图片/截图/截图 2024-12-09 09-25-36.png");
    if (img.empty()) {
        std::clog<<"Fail to imread\n";
        exit(1);
    }
    Mat grayImg;
    cv::cvtColor(img,grayImg,cv::COLOR_BGR2GRAY);
    cv::imshow("image",img);
    cv::setMouseCallback("image",getColor_gray,&grayImg);
    cv::waitKey(0);
}

void Recognize::test_E_() {
    Mat img=cv::imread("/home/lenovo/图片/截图/截图 2024-12-07 15-11-08.png");
    if (img.empty()) {
        std::clog<<"Fail to imread\n";
        exit(1);
    }
    test_C(img);

    cv::waitKey(0);
}

void Recognize::go_video3() {
    Mat img;
    while (true) {
        video>>img;

        if (img.empty()) break;

        cv::imshow("video3",img);

        cv::waitKey(60);
    }
}

void Recognize::test_D() {
    Mat img=cv::imread("/home/lenovo/图片/截图/截图 2024-12-09 10-41-37.png");
    if (img.empty()) {
        std::clog<<"Fail to imread\n";
        exit(1);
    }
    // Mat grayImg;
    // cv::cvtColor(img,grayImg,cv::COLOR_BGR2GRAY);
    test_C(img);
    cv::waitKey(0);
}

void Recognize::test_C(const Mat& img) {
    Mat grayImg,gaussianedImg,cuttedImg;
    cv::cvtColor(img,grayImg,cv::COLOR_BGR2GRAY);
    
    cv::GaussianBlur(grayImg,gaussianedImg,cv::Size(7,7),3);  // (7,7,3)
    cv::threshold(gaussianedImg,cuttedImg,params[0],255,cv::THRESH_BINARY);  // 修改了 127 -> 150

    std::vector<std::vector<cv::Point>>contours;
    cv::findContours(cuttedImg,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);  // add cv

    Mat canvas(img.rows,img.cols,img.type(),cv::Scalar(255,255,255));
    cv::drawContours(canvas,contours,-1,cv::Scalar(0,0,0),2);
    cv::imshow("canvas",canvas);

    std::vector<cv::Rect>rects;

    // std::cout<<"contours.size() = "<<contours.size()<<std::endl;

    for (auto& contour:contours) {
        cv::Rect rect=cv::boundingRect(contour);

        Mat rectMat=img(rect);
        Mat rectGrayImg=grayImg(rect);  // grayImg -> 高斯
        Mat _rectMat,_rectGrayImg,mask_A;

        cv::compare(rectGrayImg,cv::Scalar(params[1]),mask_A,cv::CMP_GT);

        rectMat.copyTo(_rectMat,mask_A);
        rectGrayImg.copyTo(_rectGrayImg,mask_A);

        if (count_gray_average(_rectGrayImg) >= params[2]) { /* 150 */

            auto special_average=special_BGR_average(_rectMat,mask_A);
            // cv::rectangle(img,rect,cv::Scalar(0,255,0),3);

            // std::cout<<std::get<0>(special_average)<<" "<<std::get<1>(special_average)<<" "<<std::get<2>(special_average)<<"\n";

            if (isColor(special_average)) {
                // cv::rectangle(img,rect,cv::Scalar(0,255,0),3);

                rects.push_back(rect);
            }

            // std::cout<<count_gray_average(_rectGrayImg)<<std::endl;
        }
    }

    const double HEIGHT_LIGHT=57.0;

    const double HEIGHT_BOARD=125.0;
    const double WIDTH_BOARD=135.0;

    const double HERO_WIDTH_BOARD=230.0;
    const double HERO_HEIGHT_BOARD=127.0;

    // std::cout<<"rects.size() = "<<rects.size()<<std::endl;

    if (rects.size()>0) {
        for (int i=0;i<rects.size()-1;++i) {
            for (int j=i+1;j<rects.size();++j) {
                if (can_tolerate_area(rects[i],rects[j]) && can_tolerate_height(rects[i],rects[j]) &&
                     height_greater_than_width(rects[i]) && height_greater_than_width(rects[j])) {

                    double height_average=static_cast<double>(rects[i].height+rects[j].height)/2;

                    cv::Point2d i_middle=get_middle(rects[i]),j_middle=get_middle(rects[j]);
                    double distance=count_distance(i_middle,j_middle);

                    double rate=distance/height_average;

                    double rate_standard=WIDTH_BOARD/HEIGHT_LIGHT;
                    double hero_rate_standard=HERO_WIDTH_BOARD/HEIGHT_LIGHT;
                    // cv::rectangle(img,rects[i],cv::Scalar(255,0,0),3);
                    // cv::rectangle(img,rects[j],cv::Scalar(255,0,0),3);

                    // std::cout<<rate/rate_standard<<std::endl;

                    // || can_tolerate_rate_hero(rate,hero_rate_standard)
                    if (can_tolerate_rate(rate,rate_standard)) {
                        // cv::rectangle(img,rects[i],cv::Scalar(0,0,255),3);
                        // cv::rectangle(img,rects[j],cv::Scalar(0,0,255),3);

                        // std::cout<<rate<<" "<<rate_standard<<std::endl;
                        // std::cout<<rate/rate_standard<<std::endl;
                    
                        double board_height=HEIGHT_BOARD/HEIGHT_LIGHT*height_average;

                        cv::Point2d rectVertices[4]={
                            cv::Point2d(i_middle.x,i_middle.y+board_height/2),
                            cv::Point2d(i_middle.x,i_middle.y-board_height/2),
                            cv::Point2d(j_middle.x,j_middle.y+board_height/2),
                            cv::Point2d(j_middle.x,j_middle.y-board_height/2)
                        };

                        double board_width=distance;
                        // can_tolerate_dist(board_width,board_height) || can_tolerate_dist_hero(board_width,board_height)

                        double big_angle=calculateAngle(rectVertices[1],rectVertices[0],rectVertices[2]);

                        // std::cout<<"big_angle = "<<big_angle<<std::endl;

                        if (can_tolerate_angle(big_angle,110.0)) {
                            cv::rectangle(img,rects[i],cv::Scalar(0,255,0),3);
                            cv::rectangle(img,rects[j],cv::Scalar(0,255,0),3);

                            cv::line(img,rectVertices[0],rectVertices[1],cv::Scalar(0,255,0),2);
                            cv::line(img,rectVertices[1],rectVertices[3],cv::Scalar(0,255,0),2);
                            cv::line(img,rectVertices[2],rectVertices[3],cv::Scalar(0,255,0),2);
                            cv::line(img,rectVertices[2],rectVertices[0],cv::Scalar(0,255,0),2);
                        }
                    } else {
                        // cv::rectangle(img,rects[i],cv::Scalar(0,0,255),3);
                        // cv::rectangle(img,rects[j],cv::Scalar(0,0,255),3);
                    }
                } else {
                    // cv::rectangle(img,rects[i],cv::Scalar(0,0,255),3);
                    // cv::rectangle(img,rects[j],cv::Scalar(0,0,255),3);
                }
            }
        }
    } else {
        // std::cout<<"it is empty\n";
    }

    cv::imshow("result",img);
}

bool Recognize::can_tolerate_rate(const double a,const double b) {
    assert(b!=0.0);

    double tolerance = params[6];  // 0.5 -> 1.8
    double rate=a/b;

    return rate>=1.0-tolerance && rate<=1.0+tolerance;
}

bool Recognize::can_tolerate_rate_hero(const double a,const double b) {
    assert(b!=0.0);

    double tolerance=0.18;
    double rate=a/b;

    return rate>=1-tolerance && rate<=1+tolerance;
}

bool Recognize::can_tolerate_area(const Rect& a,const Rect& b) {
    double areaTolerance = params[3];
    double rate=(double)a.area()/(double)b.area();

    return rate>=1-areaTolerance && rate<=1+areaTolerance;
}

bool Recognize::can_tolerate_height(const Rect& a,const Rect& b) {
    double heightTolerance=params[4];
    double rate=static_cast<double>(a.height)/b.height;

    return rate>=1-heightTolerance && rate<=1+heightTolerance;
}

bool Recognize::height_greater_than_width(const Rect& rect) {
    double tolerance=params[5];
    double rate=static_cast<double>(rect.height)/rect.width;

    return rate>=tolerance;
}

bool Recognize::can_tolerate_dist(const double bigger,const double smaller) {
    // double tolerance=params[6];
    // double rate=bigger/smaller;

    // return rate>=1 && rate<=tolerance;
}

bool Recognize::can_tolerate_dist_hero(const double bigger,const double smaller) {
    double tolerance=2.0;
    double rate=bigger/smaller;

    return rate>=1.50 && rate<=1+tolerance;
}

bool Recognize::can_tolerate_angle(const double big_angle,const double base) {
    double tolerance=params[7];  // 33.0
    double gap=std::abs(big_angle-base);

    return gap<=tolerance;
}

double Recognize::angle(const cv::Point2d& p1,const cv::Point2d& p2) {
    double dx = p2.x - p1.x;
    double dy = p2.y - p1.y;

    // 处理垂直直线的特殊情况（dx为0时斜率不存在）
    if (std::abs(dx) < 1e-6) {  // 考虑浮点数比较的精度问题，使用一个较小的阈值判断
        return 90.0;
    }

    double slope = dy / dx;
    double radian = std::atan(slope);  // 先得到弧度值

    // 将弧度转换为角度（180度对应PI弧度）
    double degree = radian * 180.0 / CV_PI;

    // 调整角度范围到(0, 180]
    if (degree < 0) {
        degree += 180.0;
    } else if (degree == 0 && dy < 0) {
        degree = 180.0;
    }

    return degree;
}

double Recognize::calculateAngle(const cv::Point2d& pointA, const cv::Point2d& pointB, const cv::Point2d& pointC) {
    // 计算线段AB的倾斜角
    double angleAB = angle(pointA, pointB);
    // 计算线段BC的倾斜角
    double angleBC = angle(pointB, pointC);

    // 通过两个倾斜角的差值计算夹角，取绝对值保证角度为正值
    double includedAngle = std::abs(angleAB - angleBC);
    // 考虑到差值可能超过180度的情况，用360度减去差值来得到正确的夹角（小于等于180度）
    if (includedAngle > 180.0) {
        includedAngle = 360.0 - includedAngle;
    }

    return includedAngle;
}

cv::Point2d Recognize::get_middle(const cv::Rect& rect) {
    double center_x=(double)rect.x+(double)rect.width/2;
    double center_y=(double)rect.y+(double)rect.height/2;

    return cv::Point2d(center_x,center_y);
}

double Recognize::count_distance(const cv::Point2d& a,const cv::Point2d& b) {
    double x=std::pow(a.x-b.x,2);
    double y=std::pow(a.y-b.y,2);

    return std::pow(x+y,0.5);
}

void Recognize::test_C_(int rate) {
    Mat frame;

    while (true) {
        this->video>>frame;

        if (frame.empty()) break;
        else {
            test_C(frame);
            cv::waitKey(rate);
        }
    }
}

BGR Recognize::count_BGR_average(const Mat& image) {
    std::vector<Mat>channels;
    cv::split(image,channels);

    if (channels.size()!= 3) {
        std::cerr << "通道分离失败，不符合彩色图像预期的通道数量" << std::endl;
        return std::make_tuple(0.0, 0.0, 0.0);
    }

    Mat blue_channel=channels[0];
    Mat green_channel=channels[1];
    Mat red_channel=channels[2];
    auto blue_mean=cv::mean(blue_channel)[0];
    auto green_mean=cv::mean(green_channel)[0];
    auto red_mean=cv::mean(red_channel)[0];

    return make_BGR(blue_mean,green_mean,red_mean);
}

BGR Recognize::special_BGR_average(const Mat& image,const Mat& mask) {
    if (image.type()!=CV_8UC3) {
        std::cout<<"不是三通道\n";
        return std::make_tuple(0.0,0.0,0.0);
    }

    if (mask.type()!=CV_8UC1) {
        std::cout<<"不是单通道\n";
        return std::make_tuple(0.0,0.0,0.0);
    }

    if (image.rows!=mask.rows || image.cols!=mask.cols) {
        std::cerr << "图像和掩码尺寸不匹配，无法进行计算" << std::endl;
        return std::make_tuple(0.0,0.0,0.0);
    }

    double sumBlue = 0.0;
    double sumGreen = 0.0;
    double sumRed = 0.0;
    int count = 0;

    for (int row = 0; row < image.rows; ++row) {
        for (int col = 0; col < image.cols; ++col) {
            if ((int)mask.at<uchar>(row, col) == 255) {
                cv::Vec3b pixel = image.at<cv::Vec3b>(row, col);
                sumBlue += pixel[0];
                sumGreen += pixel[1];
                sumRed += pixel[2];
                count++;
            }
        }
    }

    double averageBlue = (count > 0)? sumBlue / count : 0.0;
    double averageGreen = (count > 0)? sumGreen / count : 0.0;
    double averageRed = (count > 0)? sumRed / count : 0.0;

    return make_BGR(averageBlue, averageGreen, averageRed);
}

bool Recognize::isYellow(const cv::Scalar& color) {
    return color[0]<=100 && color[1]>=250 && color[2]>=250;
}

bool Recognize::isYellow(const BGR& color) {
    return std::get<0>(color)<=100 && std::get<1>(color)>=180 && std::get<2>(color) >= 180;
}

bool Recognize::isColor(const BGR& color) {
    // return std::get<0>(color)>=200 && std::get<1>(color)>=200 && std::get<2>(color) >= 200;
    return true;
}

double Recognize::count_gray_average(const Mat& image) {
    if (image.channels()!=1) {
        std::cerr << "通道分离失败，不符合彩色图像预期的通道数量" << std::endl;
        return 0;
    }

    auto meanValue=cv::mean(image);
    return meanValue[0];
}

static void getColor_gray(int event,int x,int y,int flags,void* frame) {
    if (event!=cv::EVENT_LBUTTONDOWN) {
        return;
    } else {
        Mat* img=static_cast<Mat*>(frame);

        std::cout<<(int)img->at<uchar>(y,x)<<std::endl;
    }
}

static void getColor_BGR(int event,int x,int y,int flags,void* frame) {
    if (event!=cv::EVENT_LBUTTONDOWN) {
        return;
    } else {
        Mat* img=static_cast<Mat*>(frame);
        cv::Vec3b pixel=img->at<cv::Vec3b>(y,x);

        std::cout<<(int)pixel[0]<<","<<(int)pixel[1]<<","<<(int)pixel[2]<<std::endl;
    }
}

void Recognize::loadModel() {
    String model_path="../model/svm_model.xml";

    this->loadedSvm=cv::ml::SVM::load(model_path);

    if (this->loadedSvm.empty()) {
        std::clog<<"加载模型失败\n";
        exit(1);
    } else {
        std::cout<<"加载模型成功\n";
    }
}

int Recognize::make_predict(const Mat& img) {
    auto sample=extractFeatures(img);

    return loadedSvm->predict(sample);
}

void Recognize::loadParam(int param) {
    std::vector<double>all_params(24,0.0);

    std::ifstream infile_A("../params/params.txt");

    for (int i=0;i<24;++i) {
        infile_A>>all_params[i];
    }

    for (int i=8*(param-1);i<8*param;++i) {
        this->params.push_back(all_params[i]);
    }

    infile_A.close();

    std::vector<String>video_path(3);
    
    std::ifstream infile_B("../params/path.txt");

    infile_B>>video_path[0];
    infile_B>>video_path[1];
    infile_B>>video_path[2];

    video.open(video_path[param-1]);

    if (!video.isOpened()) {
        std::cout<<"视频打开失败\n";
        exit(1);
    }
    std::cout<<"视频 = "<<video_path[param-1]<<std::endl;

    infile_B.close();

    std::cout<<"参数 = ";
    std::for_each(params.begin(),params.end(),[](const double param) {
        std::cout<<param<<" ";
    });
    std::cout<<std::endl;
}
