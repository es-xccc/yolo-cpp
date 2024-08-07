#include "utils.hpp"

chrono::steady_clock::time_point start_total_time;

int main(const int argc, const char** argv) {
    dpuOpen();
    DPUKernel *kernel = dpuLoadKernel(YOLOKERNEL);
    DPUTask *task = dpuCreateTask(kernel, 0);
    int sh = dpuGetInputTensorHeight(task, INPUTNODE);
    int sw = dpuGetInputTensorWidth(task, INPUTNODE);

    // Create a VideoCapture object to access the camera
    cv::VideoCapture cap("/dev/video0");
    if (!cap.isOpened()) {
        cerr << "Error: Unable to access the camera." << endl;
        return -1;
    }

    // Define the target resolution
    int newWidth = 640;
    int newHeight = 480;
    int frame_counter = 0;
    std::chrono::duration<double> total_duration(0);
    start_total_time = chrono::steady_clock::now();

    while (true) {
        // Read a frame from the camera
        cv::Mat frame;
        cap.read(frame);
    
        if (frame.empty()) {
            cerr << "Error: Empty frame received." << endl;
            break;
        }
    
        // Resize the frame to the new width and height
        cv::resize(frame, frame, cv::Size(newWidth, newHeight));
    
        chrono::steady_clock::time_point start_preprocessing_time = chrono::steady_clock::now();
        set_input_image(task, frame, INPUTNODE);
        chrono::steady_clock::time_point end_preprocessing_time = chrono::steady_clock::now();
        chrono::duration<double> preprocessing_duration = end_preprocessing_time - start_preprocessing_time;

        chrono::steady_clock::time_point start_dpuTask_time = chrono::steady_clock::now();
        dpuRunTask(task);
        chrono::steady_clock::time_point end_dpuTask_time = chrono::steady_clock::now();
        chrono::duration<double> dpuTask_duration = end_dpuTask_time - start_dpuTask_time;

        chrono::steady_clock::time_point start_deal_time = chrono::steady_clock::now();
        deal(task, frame, sw, sh);
        chrono::steady_clock::time_point end_deal_time = chrono::steady_clock::now();
        chrono::duration<double> deal_duration = end_deal_time - start_deal_time;
    

        cout << "preprocessing time: " << preprocessing_duration.count() << " seconds" << endl;
        cout << "DPU task time: " << dpuTask_duration.count() << " seconds" << endl;
        cout << "deal time: " << deal_duration.count() << " seconds" << endl;

        double per_process_fps = 1 / (preprocessing_duration.count() + dpuTask_duration.count() + deal_duration.count());
        cout << "per process FPS: " << per_process_fps << endl;
        cout << endl;

        chrono::steady_clock::time_point end_total_time = chrono::steady_clock::now();
        chrono::duration<double> total_duration = end_total_time - start_total_time;
        frame_counter++;
        double average_fps = frame_counter / total_duration.count();
        cout << "average FPS: " << average_fps << endl;
        cout << endl;

    }

    cap.release();
    dpuDestroyTask(task);
    dpuDestroyKernel(kernel);
    dpuClose();
    return 0;
}