#include "caffe/caffe.hpp"
#include <string>
#include <vector>

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif // USE_OPENCV

// 用于计时
#include <boost/date_time/posix_time/posix_time.hpp>

#define INPUT_W 640
#define INPUT_H 640
#define IsPadding 1
#define NUM_CLASS 1
#define NMS_THRESH 0.6
#define CONF_THRESH 0.3
std::string prototxt_path = "../model/yolov5s-4.0-focus.prototxt";
std::string caffemodel_path = "../model/yolov5s-4.0-focus.caffemodel";
std::string pic_path = "/home/willer/calibration_data/2ad80d25-b022-3b9d-a46f-853f112c2dfe.jpg";

using namespace cv;
using namespace std;
using namespace caffe;
using std::string;

using caffe::Blob;
using caffe::Caffe;
using caffe::Layer;
using caffe::Net;
using caffe::shared_ptr;
using caffe::string;
using caffe::vector;
using std::cout;
using std::endl;
using std::ostringstream;

struct Bbox{
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    float score;
    int cid;
};

struct Anchor{
    float width;
    float height;
};

std::vector<Anchor> initAnchors(){
    std::vector<Anchor> anchors;
    Anchor anchor;
    // 10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90,  156,198,  373,326
    anchor.width = 10;
    anchor.height = 13;
    anchors.emplace_back(anchor);
    anchor.width = 16;
    anchor.height = 30;
    anchors.emplace_back(anchor);
    anchor.width = 32;
    anchor.height = 23;
    anchors.emplace_back(anchor);
    anchor.width = 30;
    anchor.height = 61;
    anchors.emplace_back(anchor);
    anchor.width = 62;
    anchor.height = 45;
    anchors.emplace_back(anchor);
    anchor.width = 59;
    anchor.height = 119;
    anchors.emplace_back(anchor);
    anchor.width = 116;
    anchor.height = 90;
    anchors.emplace_back(anchor);
    anchor.width = 156;
    anchor.height = 198;
    anchors.emplace_back(anchor);
    anchor.width = 373;
    anchor.height = 326;
    anchors.emplace_back(anchor);
    return anchors;
}

template <typename T>
T clip(const T &n, const T &lower, const T &upper){
    return std::max(lower, std::min(n, upper));
}

template<class ForwardIterator>
inline size_t argmax(ForwardIterator first, ForwardIterator last)
{
    return std::distance(first, std::max_element(first, last));
}

void transform(const int &ih, const int &iw, const int &oh, const int &ow, std::vector<Bbox> &bboxes,
               bool is_padding) {
    if(is_padding){
        float scale = std::min(static_cast<float>(ow) / static_cast<float>(iw), static_cast<float>(oh) / static_cast<float>(ih));
        int nh = static_cast<int>(scale * static_cast<float>(ih));
        int nw = static_cast<int>(scale * static_cast<float>(iw));
        int dh = (oh - nh) / 2;
        int dw = (ow - nw) / 2;
        for (auto &bbox : bboxes){
            bbox.xmin = (bbox.xmin - dw) / scale;
            bbox.ymin = (bbox.ymin - dh) / scale;
            bbox.xmax = (bbox.xmax - dw) / scale;
            bbox.ymax = (bbox.ymax - dh) / scale;
        }
    }else{
        for (auto &bbox : bboxes){
            bbox.xmin = bbox.xmin * iw / ow;
            bbox.ymin = bbox.ymin * ih / oh;
            bbox.xmax = bbox.xmax * iw / ow;
            bbox.ymax = bbox.ymax * ih / oh;
        }
    }
}


cv::Mat renderBoundingBox(cv::Mat image, const std::vector<Bbox> &bboxes){
    for (auto it: bboxes){
        float score = it.score;
        cv::rectangle(image, cv::Point(it.xmin, it.ymin), cv::Point(it.xmax, it.ymax), cv::Scalar(255, 204,0), 3);
        cv::putText(image, std::to_string(score), cv::Point(it.xmin, it.ymin), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,204,255));
    }
    return image;
}

void nms_cpu(std::vector<Bbox> &bboxes, float threshold) {
    if (bboxes.empty()){
        return ;
    }
    // 1.之前需要按照score排序
    std::sort(bboxes.begin(), bboxes.end(), [&](Bbox b1, Bbox b2){return b1.score>b2.score;});
    // 2.先求出所有bbox自己的大小
    std::vector<float> area(bboxes.size());
    for (int i=0; i<bboxes.size(); ++i){
        area[i] = (bboxes[i].xmax - bboxes[i].xmin + 1) * (bboxes[i].ymax - bboxes[i].ymin + 1);
    }
    // 3.循环
    for (int i=0; i<bboxes.size(); ++i){
        for (int j=i+1; j<bboxes.size(); ){
            float left = std::max(bboxes[i].xmin, bboxes[j].xmin);
            float right = std::min(bboxes[i].xmax, bboxes[j].xmax);
            float top = std::max(bboxes[i].ymin, bboxes[j].ymin);
            float bottom = std::min(bboxes[i].ymax, bboxes[j].ymax);
            float width = std::max(right - left + 1, 0.f);
            float height = std::max(bottom - top + 1, 0.f);
            float u_area = height * width;
            float iou = (u_area) / (area[i] + area[j] - u_area);
            if (iou>=threshold){
                bboxes.erase(bboxes.begin()+j);
                area.erase(area.begin()+j);
            }else{
                ++j;
            }
        }
    }
}
template <typename T>
T sigmoid(const T &n) {
    return 1 / (1 + exp(-n));
}
void postProcessParall(const int height, const int width, int scale_idx, float postThres, float * origin_output, vector<int> Strides, vector<Anchor> Anchors, vector<Bbox> *bboxes)
{
    Bbox bbox;
    float cx, cy, w_b, h_b, score;
    int cid;
    const float *ptr = (float *)origin_output;
    for(unsigned long a=0; a<3; ++a){
        for(unsigned long h=0; h<height; ++h){
            for(unsigned long w=0; w<width; ++w){
                const float *cls_ptr =  ptr + 5;
                cid = argmax(cls_ptr, cls_ptr+NUM_CLASS);
                score = sigmoid(ptr[4]) * sigmoid(cls_ptr[cid]);
                if(score>=postThres){
                    cx = (sigmoid(ptr[0]) * 2.f - 0.5f + static_cast<float>(w)) * static_cast<float>(Strides[scale_idx]);
                    cy = (sigmoid(ptr[1]) * 2.f - 0.5f + static_cast<float>(h)) * static_cast<float>(Strides[scale_idx]);
                    w_b = powf(sigmoid(ptr[2]) * 2.f, 2) * Anchors[scale_idx * 3 + a].width;
                    h_b = powf(sigmoid(ptr[3]) * 2.f, 2) * Anchors[scale_idx * 3 + a].height;
                    bbox.xmin = clip(cx - w_b / 2, 0.F, static_cast<float>(INPUT_W - 1));
                    bbox.ymin = clip(cy - h_b / 2, 0.f, static_cast<float>(INPUT_H - 1));
                    bbox.xmax = clip(cx + w_b / 2, 0.f, static_cast<float>(INPUT_W - 1));
                    bbox.ymax = clip(cy + h_b / 2, 0.f, static_cast<float>(INPUT_H - 1));
                    bbox.score = score;
                    bbox.cid = cid;
                    //std::cout<< "bbox.cid : " << bbox.cid << std::endl;
                    bboxes->push_back(bbox);
                }
                ptr += 5 + NUM_CLASS;
            }
        }
    }
}
vector<Bbox> postProcess(vector<float *> origin_output, float postThres, float nmsThres) {

    vector<Anchor> Anchors = initAnchors();
    vector<Bbox> bboxes;
    vector<int> Strides = vector<int> {8, 16, 32};
    for (int scale_idx=0; scale_idx<3; ++scale_idx) {
        const int stride = Strides[scale_idx];
        const int width = (INPUT_W + stride - 1) / stride;
        const int height = (INPUT_H + stride - 1) / stride;
        //std::cout << "width : " << width << " " << "height : " << height << std::endl;
        float * cur_output_tensor = origin_output[scale_idx];
        postProcessParall(height, width, scale_idx, postThres, cur_output_tensor, Strides, Anchors, &bboxes);
    }
    nms_cpu(bboxes, nmsThres);
    return bboxes;
}

cv::Mat preprocess_img(cv::Mat& img) {
    int w, h, x, y;
    float r_w = INPUT_W / (img.cols*1.0);
    float r_h = INPUT_H / (img.rows*1.0);
    if (r_h > r_w) {
        w = INPUT_W;
        h = r_w * img.rows;
        x = 0;
        y = (INPUT_H - h) / 2;
    } else {
        w = r_h * img.cols;
        h = INPUT_H;
        x = (INPUT_W - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    return out;
}

int main()
{

    ::google::InitGoogleLogging("caffe"); //初始化日志文件,不调用会给出警告,但不会报错
    Caffe::set_mode(Caffe::GPU);
    Caffe::set_solver_rank(1); //不进行日志输出
    Net<float> caffe_net(prototxt_path, caffe::TEST, 0, nullptr);
    caffe_net.CopyTrainedLayersFrom(caffemodel_path);

    // 读入图片
    cv::Mat img = cv::imread(pic_path);
    //cv::Mat img = cv::imread("../model/21.jpg");
    CHECK(!img.empty()) << "Unable to decode image ";
    cv::Mat showImage = img.clone();

    // 图片预处理,并加载图片进入blob
    Blob<float> *input_layer = caffe_net.input_blobs()[0];
    float *input_data = input_layer->mutable_cpu_data();

    static float data[3 * INPUT_H * INPUT_W];
    cv::Mat pre_img = preprocess_img(img);
    std::cout << "preprocess_img finished!\n";
    int i = 0;
    for (int row = 0; row < INPUT_H; ++row) {
        uchar* uc_pixel = pre_img.data + row * pre_img.step;
        for (int col = 0; col < INPUT_W; ++col) {
            data[i] = (float)uc_pixel[2] / 255.0;
            data[i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
            data[i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
            uc_pixel += 3;
            ++i;
        }
    }

    memcpy((float *) (input_data),
           data, 3 * INPUT_H * INPUT_W * sizeof(float));

    //boost::posix_time::ptime start_time_ = boost::posix_time::microsec_clock::local_time(); //开始计时
    float total_time = 0;    
    //前向运算
    int nums = 50;
    for (int i = 0; i < nums; ++i){
        boost::posix_time::ptime start_time_1 = boost::posix_time::microsec_clock::local_time();
        caffe_net.Forward();
        boost::posix_time::ptime end_time_1 = boost::posix_time::microsec_clock::local_time();
        total_time += (end_time_1 - start_time_1).total_milliseconds(); 
        std::cout << "[ " << i << " ] " << (end_time_1 - start_time_1).total_milliseconds() << " ms." << std::endl;
    }

    //boost::posix_time::ptime end_time_ = boost::posix_time::microsec_clock::local_time(); //结束计时

    Blob<float> *output_layer0 = caffe_net.output_blobs()[2];
    const float *output0 = output_layer0->cpu_data();
    //cout << "output shape: " << output_layer0->shape(0) << " " << output_layer0->shape(1) << " " <<  output_layer0->shape(2) << " " <<  output_layer0->shape(3) <<  " " <<  output_layer0->shape(4) << endl;

    Blob<float> *output_layer1 = caffe_net.output_blobs()[0];
    const float *output1 = output_layer1->cpu_data();
    //cout << "371 shape: " << output_layer1->shape(0) << " " << output_layer1->shape(1) << " " <<  output_layer1->shape(2) << " " <<  output_layer1->shape(3) <<  " " <<  output_layer1->shape(4) << endl;

    Blob<float> *output_layer2 = caffe_net.output_blobs()[1];
    const float *output2 = output_layer2->cpu_data();
    //cout << "391 shape: " << output_layer2->shape(0) << " " << output_layer2->shape(1) << " " <<  output_layer2->shape(2) << " " <<  output_layer2->shape(3) <<  " " <<  output_layer2->shape(4) << endl;

    vector<float *> cur_output_tensors;
    cur_output_tensors.push_back(const_cast<float *>(output0));
    cur_output_tensors.push_back(const_cast<float *>(output1));
    cur_output_tensors.push_back(const_cast<float *>(output2));

    vector<Bbox> bboxes = postProcess(cur_output_tensors, CONF_THRESH, NMS_THRESH);

    transform(showImage.rows, showImage.cols, INPUT_W, INPUT_H, bboxes, IsPadding);

    showImage = renderBoundingBox(showImage, bboxes);
    cv::imwrite("reslut.jpg", showImage);


    std::cout << "average time : " << total_time / nums*1.0 << " ms" << std::endl;
}
