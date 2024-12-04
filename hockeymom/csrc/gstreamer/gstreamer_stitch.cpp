#include <gst/gst.h>
#include <nvbufsurface.h>
#include <nvds_meta.h>
#include <opencv2/opencv.hpp>
//#include "superpoint_model.h"  // Assume SuperPoint model wrapper is implemented in this file

extern "C" gboolean gst_nvdsplugin_init(GstPlugin *plugin);

static GstFlowReturn process_frame(GstBuffer *buffer1, GstBuffer *buffer2) {
    NvBufSurface *surface1 = nullptr;
    NvBufSurface *surface2 = nullptr;

    if (!gst_buffer_map(buffer1, (GstMapInfo *)&surface1, GST_MAP_READ) ||
        !gst_buffer_map(buffer2, (GstMapInfo *)&surface2, GST_MAP_READ)) {
        return GST_FLOW_ERROR;
    }

    // Convert NvBufSurface to OpenCV Mat
    cv::Mat img1 = NvBufSurfaceToCvMat(surface1);
    cv::Mat img2 = NvBufSurfaceToCvMat(surface2);

    // Run SuperPoint model to extract keypoints and descriptors
    SuperPointModel superpoint;
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;

    superpoint.extractKeypointsAndDescriptors(img1, keypoints1, descriptors1);
    superpoint.extractKeypointsAndDescriptors(img2, keypoints2, descriptors2);

    // Match keypoints between two images
    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);

    // Find homography and warp images to stitch
    std::vector<cv::Point2f> points1, points2;
    for (const auto &match : matches) {
        points1.push_back(keypoints1[match.queryIdx].pt);
        points2.push_back(keypoints2[match.trainIdx].pt);
    }

    cv::Mat homography = cv::findHomography(points1, points2, cv::RANSAC);
    cv::Mat stitched_image;
    cv::warpPerspective(img1, stitched_image, homography, cv::Size(img1.cols + img2.cols, img1.rows));

    // Copy img2 onto the stitched image
    img2.copyTo(stitched_image(cv::Rect(0, 0, img2.cols, img2.rows)));

    // Here, we would convert stitched_image back to NvBufSurface and pass it along in the pipeline
    
    gst_buffer_unmap(buffer1, (GstMapInfo *)&surface1);
    gst_buffer_unmap(buffer2, (GstMapInfo *)&surface2);

    return GST_FLOW_OK;
}

static gboolean plugin_init(GstPlugin *plugin) {
    // Register the DeepStream plugin here
    return gst_element_register(plugin, "nvds-superpoint-stitch", GST_RANK_NONE, GST_TYPE_BASE_TRANSFORM);
}

GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nvds_superpoint_stitch,
    "DeepStream plugin to stitch two video streams using SuperPoint",
    plugin_init,
    "1.0",
    "LGPL",
    "NVIDIA DeepStream",
    "https://developer.nvidia.com/deepstream-sdk"
)
