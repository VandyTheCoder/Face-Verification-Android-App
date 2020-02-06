#include <jni.h>
#include <string>
#include <vector>
#include <android/log.h>

// OpenCV Library
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc_c.h>

// Dlib Library
#include <dlib/dnn.h>
#include <dlib/dnn/loss_abstract.h>
#include <dlib/opencv.h>
#include <dlib/opencv/cv_image.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>

using namespace dlib;
using namespace std;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

// Anet DataType
using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
        alevel0<
        alevel1<
        alevel2<
        alevel3<
        alevel4<
        max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
        input_rgb_image_sized<150>
        >>>>>>>>>>>>;

extern "C" JNIEXPORT jstring JNICALL
Java_com_nhean_faceverification_MainActivity_get128DFromMat(JNIEnv *env, jobject, jstring image_addr){

    // Split String to Array in C++
    std::string image_addr_path= env->GetStringUTFChars(image_addr, 0);

    std::string first_image;
    std::string second_image;
    std::string delimiter = ",";
    size_t pos = 0;
    while ((pos = image_addr_path.find(delimiter)) != std::string::npos) {
        first_image = image_addr_path.substr(0, pos);
        __android_log_print(ANDROID_LOG_DEBUG, "native-lib :: ", "First Image %s", first_image.c_str());
        image_addr_path.erase(0, pos + delimiter.length());
    }
    second_image = image_addr_path;
    __android_log_print(ANDROID_LOG_DEBUG, "native-lib :: ", "Second Image %s", second_image.c_str());

    // Convert Mat Address(String to Long)
    std::string::size_type sz;
    long first_image_addr = std::stol (first_image,&sz);
    long second_image_addr = std::stol (second_image,&sz);
    __android_log_print(ANDROID_LOG_DEBUG, "native-lib :: ", "Done Converting(String to Long)");

    // Get First Image from Native Address
    cv::Mat &first_image_mat = *(cv::Mat *) first_image_addr;
    cv::cvtColor(first_image_mat, first_image_mat, CV_RGBA2RGB);
    __android_log_print(ANDROID_LOG_DEBUG, "native-lib :: ", "Done Get First Image");

    // Get Second Image from Native Address
    cv::Mat &second_image_mat = *(cv::Mat *) second_image_addr;
    cv::cvtColor(second_image_mat, second_image_mat, CV_RGBA2RGB);
    __android_log_print(ANDROID_LOG_DEBUG, "native-lib :: ", "Done Get Second Image");

    // Init DLIB Model Detector
    frontal_face_detector detector = get_frontal_face_detector();
    shape_predictor sp;
    deserialize("/storage/emulated/0/shape_predictor_5_face_landmarks.dat") >> sp;

    // Load the DNN responsible for face recognition.
    anet_type net;
    deserialize("/storage/emulated/0/dlib_face_recognition_resnet_model_v1.dat") >> net;

    // Convert Mat to DLIB Image
    matrix<rgb_pixel> first_image_dlib;
    assign_image(first_image_dlib, cv_image<rgb_pixel>(first_image_mat));
    __android_log_print(ANDROID_LOG_DEBUG, "native-lib :: ", "Done Converting First Mat(Mat(OpenCV) to Matrix(Dlib))");

    matrix<rgb_pixel> second_image_dlib;
    assign_image(second_image_dlib, cv_image<rgb_pixel>(second_image_mat));
    __android_log_print(ANDROID_LOG_DEBUG, "native-lib :: ", "Done Converting Second Mat(Mat(OpenCV) to Matrix(Dlib))");

    // Detecting Face
    // ? Implement OPENCV Instead
    std::vector<matrix<rgb_pixel>> faces;
    try {
        for (auto face : detector(first_image_dlib)){
            auto shape = sp(first_image_dlib, face);
            matrix<rgb_pixel> face_chip;
            extract_image_chip(first_image_dlib, get_face_chip_details(shape,150,0.25), face_chip);
            faces.push_back(move(face_chip));
            __android_log_print(ANDROID_LOG_DEBUG, "native-lib :: ", "Done Getting Face Chip From Frist Image");
        }
        for (auto face : detector(second_image_dlib)){
            auto shape = sp(second_image_dlib, face);
            matrix<rgb_pixel> face_chip;
            extract_image_chip(second_image_dlib, get_face_chip_details(shape,150,0.25), face_chip);
            faces.push_back(move(face_chip));
            __android_log_print(ANDROID_LOG_DEBUG, "native-lib :: ", "Done Getting Face Chip From Second Image");
        }
    } catch (const std::exception& e) {
        __android_log_print(ANDROID_LOG_DEBUG, "native-lib :: ", "Error: %s", e.what());
    }

    __android_log_print(ANDROID_LOG_DEBUG, "native-lib :: ", "Number of Face %d", faces.size());

    double distance = 0.0;
    try {
        std::vector<matrix<float,0,1>> face_descriptors = net(faces);
        distance= dlib::length(face_descriptors[0] - face_descriptors[1]);
        __android_log_print(ANDROID_LOG_DEBUG, "native-lib :: ", "Distance of the Face: %f", distance);

    } catch (const std::exception& e) {
        __android_log_print(ANDROID_LOG_DEBUG, "native-lib :: ", "Error: %s", e.what());
    }

    std::string distance_of_image = std::to_string(distance);
    return env->NewStringUTF(distance_of_image.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_nhean_faceverification_MainActivity_stringFromJNI(JNIEnv *env, jobject /* this */) {
    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
}