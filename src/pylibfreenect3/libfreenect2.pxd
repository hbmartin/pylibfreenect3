# distutils: language = c++

from libc.stdint cimport uint16_t, uint32_t
from libcpp cimport bool
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "libfreenect2/frame_listener.hpp" namespace "libfreenect2":
    cdef enum NativeFrameType "libfreenect2::Frame::Type":
        FRAME_COLOR "libfreenect2::Frame::Color"
        FRAME_IR "libfreenect2::Frame::Ir"
        FRAME_DEPTH "libfreenect2::Frame::Depth"

    cdef enum NativeFrameFormat "libfreenect2::Frame::Format":
        FORMAT_INVALID "libfreenect2::Frame::Invalid"
        FORMAT_RAW "libfreenect2::Frame::Raw"
        FORMAT_FLOAT "libfreenect2::Frame::Float"
        FORMAT_BGRX "libfreenect2::Frame::BGRX"
        FORMAT_RGBX "libfreenect2::Frame::RGBX"
        FORMAT_GRAY "libfreenect2::Frame::Gray"

    cdef cppclass NativeFrame "libfreenect2::Frame":
        size_t width
        size_t height
        size_t bytes_per_pixel
        unsigned char *data
        uint32_t timestamp
        uint32_t sequence
        float exposure
        float gain
        float gamma
        uint32_t status
        NativeFrameFormat format
        NativeFrame(size_t, size_t, size_t, unsigned char *) except +

    cdef cppclass NativeFrameListener "libfreenect2::FrameListener":
        pass

cdef extern from "libfreenect2/frame_listener_impl.h" namespace "libfreenect2":
    cdef cppclass NativeSyncListener "libfreenect2::SyncMultiFrameListener":
        NativeSyncListener(unsigned int) except +
        bool hasNewFrame() except +
        void waitForNewFrame(map[NativeFrameType, NativeFrame *] &) except + nogil
        bool waitForNewFrame(map[NativeFrameType, NativeFrame *] &, int) except + nogil
        void release(map[NativeFrameType, NativeFrame *] &) except + nogil

cdef extern from "libfreenect2/color_settings.h" namespace "libfreenect2":
    cdef enum NativeColorSetting "libfreenect2::ColorSettingCommandType":
        pass

cdef extern from "libfreenect2/led_settings.h" namespace "libfreenect2":
    cdef cppclass NativeLedSettings "libfreenect2::LedSettings":
        uint16_t LedId
        uint16_t Mode
        uint16_t StartLevel
        uint16_t StopLevel
        uint32_t IntervalInMs
        uint32_t Reserved

cdef extern from "libfreenect2/packet_pipeline.h" namespace "libfreenect2":
    cdef cppclass NativePacketPipeline "libfreenect2::PacketPipeline":
        const string &getName() const
        bool good() const

    cdef cppclass NativeDumpPipeline "libfreenect2::DumpPacketPipeline"(NativePacketPipeline):
        const unsigned char *getDepthP0Tables(size_t *)
        const float *getDepthXTable(size_t *)
        const float *getDepthZTable(size_t *)
        const short *getDepthLookupTable(size_t *)

    NativePacketPipeline *createPacketPipeline(const string &, int) except + nogil
    NativePacketPipeline *createDefaultPacketPipeline() except + nogil
    vector[string] getCompiledPacketPipelines() except + nogil
    vector[string] getAvailablePacketPipelines() except + nogil

cdef extern from "libfreenect2/libfreenect2.hpp" namespace "libfreenect2":
    string getVersion() except +
    uint32_t getApiVersion() except +
    string getBuildRevision() except +

    cdef cppclass NativeDevice "libfreenect2::Freenect2Device":
        cppclass ColorCameraParams:
            float fx, fy, cx, cy
            float shift_d, shift_m
            float mx_x3y0, mx_x0y3, mx_x2y1, mx_x1y2, mx_x2y0
            float mx_x0y2, mx_x1y1, mx_x1y0, mx_x0y1, mx_x0y0
            float my_x3y0, my_x0y3, my_x2y1, my_x1y2, my_x2y0
            float my_x0y2, my_x1y1, my_x1y0, my_x0y1, my_x0y0

        cppclass IrCameraParams:
            float fx, fy, cx, cy, k1, k2, k3, p1, p2

        cppclass Config:
            float MinDepth
            float MaxDepth
            bool EnableBilateralFilter
            bool EnableEdgeAwareFilter
            Config() except +

        string getSerialNumber() except +
        string getFirmwareVersion() except +
        string getPacketPipelineName() except +
        ColorCameraParams getColorCameraParams() except +
        IrCameraParams getIrCameraParams() except +
        void setColorCameraParams(const ColorCameraParams &) except + nogil
        void setIrCameraParams(const IrCameraParams &) except + nogil
        void setConfiguration(const Config &) except + nogil
        void setColorFrameListener(NativeFrameListener *) except + nogil
        void setIrAndDepthFrameListener(NativeFrameListener *) except + nogil
        void setColorAutoExposure(float) except + nogil
        void setColorSemiAutoExposure(float) except + nogil
        void setColorManualExposure(float, float) except + nogil
        void setColorSetting(NativeColorSetting, uint32_t) except + nogil
        void setColorSetting(NativeColorSetting, float) except + nogil
        uint32_t getColorSetting(NativeColorSetting) except + nogil
        float getColorSettingFloat(NativeColorSetting) except + nogil
        void setLedStatus(NativeLedSettings) except + nogil
        bool start() except + nogil
        bool startStreams(bool, bool) except + nogil
        bool stop() except + nogil
        bool close() except + nogil

    cdef cppclass NativeFreenect2 "libfreenect2::Freenect2":
        NativeFreenect2() except +
        int enumerateDevices() except + nogil
        string getDeviceSerialNumber(int) except +
        string getDefaultDeviceSerialNumber() except +
        NativeDevice *openDevice(int) except + nogil
        NativeDevice *openDevice(int, const NativePacketPipeline *) except + nogil
        NativeDevice *openDevice(const string &) except + nogil
        NativeDevice *openDevice(const string &, const NativePacketPipeline *) except + nogil
        NativeDevice *openDefaultDevice() except + nogil
        NativeDevice *openDefaultDevice(const NativePacketPipeline *) except + nogil

    cdef cppclass ReplayCalibration "libfreenect2::Freenect2Replay::Calibration":
        NativeDevice.ColorCameraParams color
        NativeDevice.IrCameraParams ir
        vector[unsigned char] p0_tables
        vector[float] x_table
        vector[float] z_table
        vector[short] lookup_table

    cdef cppclass NativeReplay "libfreenect2::Freenect2Replay":
        NativeReplay() except +
        NativeDevice *openDevice(const vector[string] &) except + nogil
        NativeDevice *openDevice(const vector[string] &, const NativePacketPipeline *) except + nogil
        NativeDevice *openDevice(const vector[string] &, const ReplayCalibration &) except + nogil
        NativeDevice *openDevice(const vector[string] &, const ReplayCalibration &, const NativePacketPipeline *) except + nogil

cdef extern from "libfreenect2/registration.h" namespace "libfreenect2":
    cdef cppclass NativeRegistration "libfreenect2::Registration":
        NativeRegistration(NativeDevice.IrCameraParams, NativeDevice.ColorCameraParams) except +
        void apply(int, int, float, float &, float &) except + nogil
        void apply(const NativeFrame *, const NativeFrame *, NativeFrame *, NativeFrame *, bool, NativeFrame *, int *) except + nogil
        void undistortDepth(const NativeFrame *, NativeFrame *) except + nogil
        void getPointXYZRGB(const NativeFrame *, const NativeFrame *, int, int, float &, float &, float &, float &) except + nogil
        void getPointXYZ(const NativeFrame *, int, int, float &, float &, float &) except + nogil

cdef extern from "libfreenect2/logger.h" namespace "libfreenect2":
    cdef enum NativeLoggerLevel "libfreenect2::Logger::Level":
        pass

    cdef cppclass NativeLogger "libfreenect2::Logger":
        NativeLoggerLevel level() const

    NativeLoggerLevel loggerDefaultLevel "libfreenect2::Logger::getDefaultLevel"()
    string loggerLevelToString "libfreenect2::Logger::level2str"(NativeLoggerLevel)
    NativeLogger *createConsoleLogger(NativeLoggerLevel)
    NativeLogger *createConsoleLoggerWithDefaultLevel()
    NativeLogger *getGlobalLogger()
    void setGlobalLogger(NativeLogger *)
