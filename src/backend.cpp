#include <cpprest/http_listener.h>
#include <cpprest/json.h>
#include <opencv2/opencv.hpp>
#include "ConvNet.h"

using namespace web;
using namespace web::http;
using namespace web::http::experimental::listener;


void handle_options(http_request request) {
    http_response response(status_codes::OK);
    response.headers().add("Access-Control-Allow-Origin", "*");
    response.headers().add("Access-Control-Allow-Methods", "POST, OPTIONS");
    response.headers().add("Access-Control-Allow-Headers", "Content-Type");
    request.reply(response);
}

void handle_post(http_request request) {
    auto content_type = request.headers().content_type();

    // Check if the content type is "image/jpeg" or similar
    if (content_type == "image/jpeg" || content_type == "image/png") {
        request.extract_vector().then([=](std::vector<unsigned char> body_data) {
            cudaSetDevice(0); // Ensure CUDA context in this thread

            ConvNet network(28, 28, 5, 5, 6, 10); // Create a fresh instance for each request
            network.loadWeights(
                "/home/mzingerenko/Desktop/LecturesProject_Example/data/conv_kernels.txt",
                "/home/mzingerenko/Desktop/LecturesProject_Example/data/fc_weights.txt",
                "/home/mzingerenko/Desktop/LecturesProject_Example/data/fc_biases.txt");

            cv::Mat img = cv::imdecode(body_data, cv::IMREAD_COLOR);

            if (img.empty()) {
                http_response response(status_codes::BadRequest);
                response.headers().add("Access-Control-Allow-Origin", "*");
                response.set_body(U("Invalid image data."));
                request.reply(response);
                return;
            }

            std::vector<float> input = network.preprocessImage(img);

            // Debugging logs
            std::cout << "Input sum: " << std::accumulate(input.begin(), input.end(), 0.0f) << std::endl;

            std::vector<float> predictions;
            network.predict(input, predictions);

            // Synchronize CUDA
            cudaDeviceSynchronize();

            // Debugging logs
            std::cout << "Raw predictions: ";
            for (float val : predictions) {
                std::cout << val << " ";
            }
            std::cout << std::endl;

            json::value response_json = json::value::object();
            for (size_t i = 0; i < predictions.size(); ++i) {
                response_json[U("Class_" + std::to_string(i))] = json::value::number(predictions[i]);
            }

            http_response response(status_codes::OK);
            response.headers().add("Access-Control-Allow-Origin", "*");
            response.headers().add("Content-Type", "application/json");
            response.set_body(response_json);
            request.reply(response);
        }).wait(); // Ensure completion of async operation
    } else {
        http_response response(status_codes::UnsupportedMediaType);
        response.headers().add("Access-Control-Allow-Origin", "*");
        response.set_body(U("Unsupported content type. Please upload an image."));
        request.reply(response);
    }
}

int main() {
    http_listener listener(U("http://localhost:8080/predict"));

    listener.support(methods::POST, handle_post);
    listener.support(methods::OPTIONS, handle_options);  // Support CORS preflight requests

    try {
        listener.open().wait();
        std::cout << "Server started at http://localhost:8080/predict" << std::endl;
        std::string line;
        std::getline(std::cin, line); // Keep the server running
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}
