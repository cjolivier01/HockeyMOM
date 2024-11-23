#include <gst/gst.h>
#include <iostream>

int main(int argc, char *argv[]) {
    // Initialize GStreamer
    gst_init(&argc, &argv);

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <Video File Path>\n";
        return -1;
    }

    // Create the elements
    GstElement *pipeline = gst_pipeline_new("video-audio-player");
    GstElement *source = gst_element_factory_make("filesrc", "file-source");
    GstElement *demuxer = gst_element_factory_make("qtdemux", "demuxer");
    GstElement *decoder_audio = gst_element_factory_make("avdec_aac", "audio-decoder");
    GstElement *decoder_video = gst_element_factory_make("avdec_h264", "video-decoder");
    GstElement *audioconv = gst_element_factory_make("audioconvert", "audio-converter");
    GstElement *audiosink = gst_element_factory_make("autoaudiosink", "audio-output");
    GstElement *videosink = gst_element_factory_make("autovideosink", "video-output");

    if (!pipeline || !source || !demuxer || !decoder_audio || !decoder_video || !audioconv || !audiosink || !videosink) {
        std::cerr << "Not all elements could be created.\n";
        return -1;
    }

    // Set up the pipeline
    gst_bin_add_many(GST_BIN(pipeline), source, demuxer, decoder_audio, decoder_video, audioconv, audiosink, videosink, NULL);
    gst_element_link(source, demuxer);

    gst_element_link_many(decoder_audio, audioconv, audiosink, NULL);
    gst_element_link_many(decoder_video, videosink, NULL);

    g_signal_connect(demuxer, "pad-added", G_CALLBACK(on_pad_added), NULL);

    // Set the source file location
    g_object_set(G_OBJECT(source), "location", argv[1], NULL);

    // Start playing
    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    // Wait until error or EOS
    GstBus *bus = gst_element_get_bus(pipeline);
    GstMessage *msg = gst_bus_timed_pop_filtered(bus, GST_CLOCK_TIME_NONE,
                                                 GstMessageType(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));

    // Free resources
    if (msg != nullptr) {
        gst_message_unref(msg);
    }
    gst_object_unref(bus);
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);

    return 0;
}

// This function is called when a new pad is created in the demuxer
void on_pad_added(GstElement *element, GstPad *pad, gpointer data) {
    GstPad *sinkpad;
    GstElement *decoder = NULL;

    // Check the new pad's name
    gchar *pad_name = gst_pad_get_name(pad);
    if (g_str_has_prefix(pad_name, "audio")) {
        decoder = gst_bin_get_by_name(GST_BIN(data), "audio-decoder");
    } else if (g_str_has_prefix(pad_name, "video")) {
        decoder = gst_bin_get_by_name(GST_BIN(data), "video-decoder");
    }
    g_free(pad_name);

    if (decoder) {
        sinkpad = gst_element_get_static_pad(decoder, "sink");
        gst_pad_link(pad, sinkpad);
        gst_object_unref(sinkpad);
    }
}
