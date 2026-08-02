# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

from cpython.exc cimport PyErr_CheckSignals
from cpython.ref cimport Py_INCREF
from cython.operator cimport dereference as deref, preincrement as inc
from libc.stdint cimport int32_t, uint8_t, uint32_t, uint64_t
from libc.string cimport memcpy, memset
from libcpp.map cimport map
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "unistd.h":
    int c_getpid "getpid"() noexcept nogil

import math
import time
import numpy as np
cimport numpy as cnp

from . cimport libfreenect2 as lf
from .errors import (
    BackendUnavailableError,
    CalibrationError,
    DeviceOpenError,
    DeviceStateError,
    FrameTimeoutError,
    RecordingError,
)

cnp.import_array()

cdef int _WAIT_SLICE_MS = 100

# Keyed by RgbDecoder values; sourced from the native enum so the Python
# layer cannot drift from the core's PacketPipelineConfig ordering.
RGB_DECODER_VALUES = {
    "auto": <int>lf.RGB_DECODER_AUTO,
    "turbojpeg": <int>lf.RGB_DECODER_TURBOJPEG,
    "videotoolbox": <int>lf.RGB_DECODER_VIDEOTOOLBOX,
    "vaapi": <int>lf.RGB_DECODER_VAAPI,
    "tegrajpeg": <int>lf.RGB_DECODER_TEGRAJPEG,
}


cdef void _require_process(long owner_pid) except *:
    cdef long current_pid = c_getpid()
    if owner_pid != current_pid:
        raise DeviceStateError(
            "native camera resources cannot be used after fork; "
            f"create them in the current process (created in PID {owner_pid}, "
            f"current PID {current_pid})"
        )


cdef object _text(string value):
    return (<bytes>value).decode("utf-8", "replace")


cdef lf.NativeFrameType _frame_type(int value):
    if value == 1:
        return lf.FRAME_COLOR
    if value == 2:
        return lf.FRAME_IR
    if value == 4:
        return lf.FRAME_DEPTH
    raise ValueError(f"invalid frame type: {value}")


cdef void _fill_color(lf.NativeDevice.ColorCameraParams *target, object source):
    target.fx = source.fx
    target.fy = source.fy
    target.cx = source.cx
    target.cy = source.cy
    target.shift_d = source.shift_d
    target.shift_m = source.shift_m
    target.mx_x3y0 = source.mx_x3y0
    target.mx_x0y3 = source.mx_x0y3
    target.mx_x2y1 = source.mx_x2y1
    target.mx_x1y2 = source.mx_x1y2
    target.mx_x2y0 = source.mx_x2y0
    target.mx_x0y2 = source.mx_x0y2
    target.mx_x1y1 = source.mx_x1y1
    target.mx_x1y0 = source.mx_x1y0
    target.mx_x0y1 = source.mx_x0y1
    target.mx_x0y0 = source.mx_x0y0
    target.my_x3y0 = source.my_x3y0
    target.my_x0y3 = source.my_x0y3
    target.my_x2y1 = source.my_x2y1
    target.my_x1y2 = source.my_x1y2
    target.my_x2y0 = source.my_x2y0
    target.my_x0y2 = source.my_x0y2
    target.my_x1y1 = source.my_x1y1
    target.my_x1y0 = source.my_x1y0
    target.my_x0y1 = source.my_x0y1
    target.my_x0y0 = source.my_x0y0


cdef void _fill_ir(lf.NativeDevice.IrCameraParams *target, object source):
    target.fx = source.fx
    target.fy = source.fy
    target.cx = source.cx
    target.cy = source.cy
    target.k1 = source.k1
    target.k2 = source.k2
    target.k3 = source.k3
    target.p1 = source.p1
    target.p2 = source.p2


cdef int _distortion_model(object value) except -1:
    cdef str name = str(value)
    if name == "none":
        return <int>lf.DISTORTION_NONE
    if name == "brown_conrady_5":
        return <int>lf.DISTORTION_BROWN_CONRADY_5
    if name == "rational_8":
        return <int>lf.DISTORTION_RATIONAL_8
    raise ValueError(f"unknown distortion model: {value!r}")


cdef str _distortion_name(lf.NativeDistortionModel value):
    if value == lf.DISTORTION_NONE:
        return "none"
    if value == lf.DISTORTION_BROWN_CONRADY_5:
        return "brown_conrady_5"
    if value == lf.DISTORTION_RATIONAL_8:
        return "rational_8"
    raise ValueError(f"unknown native distortion model: {<int>value}")


cdef void _fill_projective_camera(lf.NativeProjectiveCameraModel *target, object source):
    cdef size_t index
    target.width = source.width
    target.height = source.height
    target.fx = source.fx
    target.fy = source.fy
    target.cx = source.cx
    target.cy = source.cy
    target.distortion_model = <lf.NativeDistortionModel>_distortion_model(source.distortion_model)
    for index in range(8):
        lf.setCameraDistortion(
            deref(target), index,
            source.distortion[index] if index < len(source.distortion) else 0.0
        )


cdef dict _projective_camera_mapping(lf.NativeProjectiveCameraModel value):
    cdef size_t count = 0
    cdef size_t index
    if value.distortion_model == lf.DISTORTION_BROWN_CONRADY_5:
        count = 5
    elif value.distortion_model == lf.DISTORTION_RATIONAL_8:
        count = 8
    return {
        "width": value.width,
        "height": value.height,
        "fx": value.fx,
        "fy": value.fy,
        "cx": value.cx,
        "cy": value.cy,
        "distortion_model": _distortion_name(value.distortion_model),
        "distortion": tuple(lf.cameraDistortion(value, index) for index in range(count)),
    }


def scale_projective_camera_model(model, uint32_t width, uint32_t height):
    cdef lf.NativeProjectiveCameraModel native
    cdef lf.NativeProjectiveCameraModel scaled
    _fill_projective_camera(&native, model)
    with nogil:
        scaled = native.scaledTo(width, height)
    return _projective_camera_mapping(scaled)


def rectify_projective_camera_model(model):
    cdef lf.NativeProjectiveCameraModel native
    cdef lf.NativeProjectiveCameraModel rectified
    _fill_projective_camera(&native, model)
    with nogil:
        rectified = native.rectified()
    return _projective_camera_mapping(rectified)


def core_version():
    return _text(lf.getVersion())


def core_api_version():
    return int(lf.getApiVersion())


def core_revision():
    return _text(lf.getBuildRevision())


def compiled_pipelines():
    cdef vector[string] values
    cdef size_t i
    with nogil:
        values = lf.getCompiledPacketPipelines()
    return frozenset(_text(values[i]) for i in range(values.size()))


def available_pipelines():
    cdef vector[string] values
    cdef size_t i
    with nogil:
        values = lf.getAvailablePacketPipelines()
    return frozenset(_text(values[i]) for i in range(values.size()))


cdef class NativePipeline:
    cdef lf.NativePacketPipeline *ptr
    cdef bint consumed
    cdef bint device_closed
    cdef object requested_name
    cdef long owner_pid

    def __cinit__(self, name, int device_id=-1, int rgb_decoder=0,
                  vaapi_device=None, bint allow_fallback=True):
        self.ptr = NULL
        self.consumed = False
        self.device_closed = False
        self.requested_name = str(name)
        self.owner_pid = c_getpid()
        cdef string encoded = str(name).encode("utf-8")
        cdef lf.NativePacketPipelineConfig config = lf.NativePacketPipelineConfig()
        config.rgb_decoder = <lf.NativeRgbDecoder>rgb_decoder
        config.vaapi_device = ("" if vaapi_device is None else str(vaapi_device)).encode("utf-8")
        config.allow_fallback = allow_fallback
        try:
            with nogil:
                if encoded == b"auto":
                    self.ptr = lf.createDefaultPacketPipeline(config)
                else:
                    self.ptr = lf.createPacketPipeline(encoded, config, device_id)
        except Exception as error:
            raise BackendUnavailableError(f"pipeline {name!r} could not be constructed") from error
        if self.ptr == NULL:
            raise BackendUnavailableError(f"pipeline {name!r} is not compiled")
        if not self.ptr.good():
            del self.ptr
            self.ptr = NULL
            raise BackendUnavailableError(f"pipeline {name!r} is not usable on this machine")

    def __dealloc__(self):
        if (self.owner_pid == c_getpid() and self.ptr != NULL and
                not self.consumed):
            del self.ptr
            self.ptr = NULL

    cdef lf.NativePacketPipeline *_consume(self) except NULL:
        _require_process(self.owner_pid)
        if self.ptr == NULL or self.consumed:
            raise DeviceStateError("packet pipelines are single-use")
        self.consumed = True
        return self.ptr

    cdef void _open_failed(self):
        self.ptr = NULL
        self.device_closed = True

    cdef void _attach(self):
        self.device_closed = False

    cdef void _device_closed(self):
        self.device_closed = True
        self.ptr = NULL

    @property
    def name(self):
        _require_process(self.owner_pid)
        if self.ptr == NULL:
            return self.requested_name
        return _text(self.ptr.getName())

    @property
    def is_consumed(self):
        _require_process(self.owner_pid)
        return self.consumed != 0

    cdef void _require_dump(self) except *:
        _require_process(self.owner_pid)
        if self.device_closed:
            raise DeviceStateError("dump tables are unavailable after the device is closed")
        if self.ptr == NULL or self.name != "dump":
            raise DeviceStateError("dump tables require a live DumpPacketPipeline")

    def depth_p0_tables(self):
        self._require_dump()
        cdef size_t length = 0
        cdef const unsigned char *data = (<lf.NativeDumpPipeline *>self.ptr).getDepthP0Tables(&length)
        if data == NULL:
            raise DeviceStateError("P0 tables are not ready")
        cdef cnp.ndarray result = np.empty(length, dtype=np.uint8)
        if length:
            memcpy(cnp.PyArray_DATA(result), data, length)
        return result

    def depth_x_table(self):
        self._require_dump()
        cdef size_t length = 0
        cdef const float *data = (<lf.NativeDumpPipeline *>self.ptr).getDepthXTable(&length)
        if data == NULL:
            raise DeviceStateError("X table is not ready")
        cdef cnp.ndarray result = np.empty(length, dtype=np.float32)
        if length:
            memcpy(cnp.PyArray_DATA(result), data, length * sizeof(float))
        return result

    def depth_z_table(self):
        self._require_dump()
        cdef size_t length = 0
        cdef const float *data = (<lf.NativeDumpPipeline *>self.ptr).getDepthZTable(&length)
        if data == NULL:
            raise DeviceStateError("Z table is not ready")
        cdef cnp.ndarray result = np.empty(length, dtype=np.float32)
        if length:
            memcpy(cnp.PyArray_DATA(result), data, length * sizeof(float))
        return result

    def depth_lookup_table(self):
        self._require_dump()
        cdef size_t length = 0
        cdef const short *data = (<lf.NativeDumpPipeline *>self.ptr).getDepthLookupTable(&length)
        if data == NULL:
            raise DeviceStateError("lookup table is not ready")
        cdef cnp.ndarray result = np.empty(length, dtype=np.int16)
        if length:
            memcpy(cnp.PyArray_DATA(result), data, length * sizeof(short))
        return result


cdef class NativeFrameSet
cdef class NativeSyncFrameListener
cdef class NativeAlignedFrameListener
cdef class NativeCalibrationProfileHandle
cdef class NativeProjectiveRegistrationHandle
cdef class NativeRecordingWriterHandle


cdef class NativeFrame:
    cdef lf.NativeFrame *ptr
    cdef bint owns_ptr
    cdef object parent
    cdef object numpy_owner
    cdef int frame_type
    cdef long owner_pid

    def __cinit__(self):
        self.ptr = NULL
        self.owns_ptr = False
        self.parent = None
        self.numpy_owner = None
        self.frame_type = -1
        self.owner_pid = c_getpid()

    def __dealloc__(self):
        if self.owner_pid == c_getpid():
            if self.owns_ptr and self.ptr != NULL:
                del self.ptr
            if self.parent is not None:
                (<NativeFrameSet>self.parent)._drop_borrow()
        self.ptr = NULL
        self.parent = None

    @staticmethod
    def allocate(size_t width, size_t height, size_t bytes_per_pixel, int frame_type=-1,
                 int frame_format=0, uint32_t timestamp=0, uint32_t sequence=0,
                 uint64_t arrival_timestamp_us=0,
                 float exposure=0.0, float gain=0.0, float gamma=0.0,
                 uint32_t status=0):
        cdef NativeFrame result = NativeFrame()
        result.ptr = new lf.NativeFrame(
            width, height, bytes_per_pixel, NULL, <lf.NativeFrameFormat>frame_format
        )
        result.owns_ptr = True
        if result.ptr.data != NULL:
            memset(result.ptr.data, 0, width * height * bytes_per_pixel)
        result.frame_type = frame_type
        result.ptr.timestamp = timestamp
        result.ptr.arrival_timestamp_us = arrival_timestamp_us
        result.ptr.sequence = sequence
        result.ptr.exposure = exposure
        result.ptr.gain = gain
        result.ptr.gamma = gamma
        result.ptr.status = status
        return result

    @staticmethod
    def from_array(object array, int frame_type=-1, int frame_format=0,
                   uint32_t timestamp=0, uint32_t sequence=0,
                   uint64_t arrival_timestamp_us=0,
                   float exposure=0.0, float gain=0.0, float gamma=0.0,
                   uint32_t status=0):
        cdef cnp.ndarray value = np.asarray(array)
        if not value.flags.c_contiguous:
            raise ValueError("frame arrays must be C-contiguous")
        if not value.flags.writeable:
            raise ValueError("frame arrays must be writable")
        if value.dtype not in (np.dtype(np.uint8), np.dtype(np.float32)):
            raise TypeError("frame arrays must use uint8 or float32")
        if frame_format == <int>lf.FORMAT_RAW:
            if value.dtype != np.uint8 or value.ndim != 1:
                raise ValueError("raw frames require a one-dimensional uint8 array")
        elif frame_format == <int>lf.FORMAT_FLOAT:
            if value.dtype != np.float32 or value.ndim != 2:
                raise ValueError("float frames require a two-dimensional float32 array")
        elif frame_format == <int>lf.FORMAT_GRAY:
            if value.dtype != np.uint8 or value.ndim != 2:
                raise ValueError("gray frames require a two-dimensional uint8 array")
        elif frame_format == <int>lf.FORMAT_BGRX or frame_format == <int>lf.FORMAT_RGBX:
            if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 4:
                raise ValueError("color frames require a (height, width, 4) uint8 array")
        else:
            raise ValueError("array-backed frames require a valid concrete format")
        cdef size_t height
        cdef size_t width
        cdef size_t bpp
        if value.ndim == 2:
            height = value.shape[0]
            width = value.shape[1]
            bpp = value.dtype.itemsize
        elif value.ndim == 3 and value.shape[2] == 4 and value.dtype == np.uint8:
            height = value.shape[0]
            width = value.shape[1]
            bpp = 4
        elif value.ndim == 1 and value.dtype == np.uint8:
            height, width, bpp = 1, 1, value.size
        else:
            raise ValueError("expected a 1-D raw, 2-D scalar, or (H, W, 4) color array")
        cdef NativeFrame result = NativeFrame()
        result.ptr = new lf.NativeFrame(
            width, height, bpp, <unsigned char *>cnp.PyArray_DATA(value),
            <lf.NativeFrameFormat>frame_format
        )
        result.owns_ptr = True
        result.numpy_owner = value
        result.frame_type = frame_type
        result.ptr.timestamp = timestamp
        result.ptr.arrival_timestamp_us = arrival_timestamp_us
        result.ptr.sequence = sequence
        result.ptr.exposure = exposure
        result.ptr.gain = gain
        result.ptr.gamma = gamma
        result.ptr.status = status
        return result

    cdef void _check(self) except *:
        _require_process(self.owner_pid)
        if self.ptr == NULL:
            raise DeviceStateError("frame is no longer valid")

    @property
    def width(self):
        self._check()
        return self.ptr.width

    @property
    def height(self):
        self._check()
        return self.ptr.height

    @property
    def bytes_per_pixel(self):
        self._check()
        return self.ptr.bytes_per_pixel

    @property
    def timestamp(self):
        self._check()
        return self.ptr.timestamp

    @property
    def arrival_timestamp_us(self):
        self._check()
        return self.ptr.arrival_timestamp_us

    @property
    def sequence(self):
        self._check()
        return self.ptr.sequence

    @property
    def exposure(self):
        self._check()
        return self.ptr.exposure

    @property
    def gain(self):
        self._check()
        return self.ptr.gain

    @property
    def gamma(self):
        self._check()
        return self.ptr.gamma

    @property
    def status(self):
        self._check()
        return self.ptr.status

    @property
    def format(self):
        self._check()
        return <int>self.ptr.format

    @property
    def type(self):
        _require_process(self.owner_pid)
        return self.frame_type

    def to_numpy(self, bint copy=False):
        self._check()
        cdef cnp.npy_intp shape[3]
        cdef cnp.ndarray array
        cdef int ndim
        cdef int typenum
        if self.ptr.format == lf.FORMAT_RAW:
            ndim = 1
            shape[0] = self.ptr.bytes_per_pixel
            typenum = cnp.NPY_UINT8
        elif self.ptr.format == lf.FORMAT_FLOAT:
            ndim = 2
            shape[0] = self.ptr.height
            shape[1] = self.ptr.width
            typenum = cnp.NPY_FLOAT32
        elif self.ptr.format == lf.FORMAT_GRAY:
            ndim = 2
            shape[0] = self.ptr.height
            shape[1] = self.ptr.width
            typenum = cnp.NPY_UINT8
        elif self.ptr.format == lf.FORMAT_BGRX or self.ptr.format == lf.FORMAT_RGBX:
            ndim = 3
            shape[0] = self.ptr.height
            shape[1] = self.ptr.width
            shape[2] = 4
            typenum = cnp.NPY_UINT8
        else:
            raise ValueError(f"unsupported frame format: {<int>self.ptr.format}")
        array = cnp.PyArray_SimpleNewFromData(ndim, shape, typenum, self.ptr.data)
        Py_INCREF(self)
        if cnp.PyArray_SetBaseObject(array, self) < 0:
            raise MemoryError("failed to attach frame lifetime to NumPy array")
        return array.copy() if copy else array

    def to_color(self, object output, int order):
        self._check()
        if not isinstance(output, np.ndarray):
            raise TypeError("output must be a NumPy array")
        cdef cnp.ndarray destination = <cnp.ndarray>output
        cdef bint ok
        cdef unsigned char *destination_ptr
        cdef size_t destination_size
        if (destination.dtype != np.dtype(np.uint8) or destination.ndim != 3 or
                destination.shape[0] != self.ptr.height or
                destination.shape[1] != self.ptr.width or destination.shape[2] != 3 or
                not destination.flags.c_contiguous or not destination.flags.writeable):
            raise ValueError("output must be a writable contiguous (height, width, 3) uint8 array")
        destination_ptr = <unsigned char *>cnp.PyArray_DATA(destination)
        destination_size = destination.nbytes
        with nogil:
            ok = lf.convertColorFrame(
                self.ptr, <lf.NativeColorOrder>order, destination_ptr, destination_size
            )
        if not ok:
            raise ValueError("frame is not a valid BGRX/RGBX color frame")
        return output


cdef NativeFrame _borrow_frame(lf.NativeFrame *ptr, int frame_type, NativeFrameSet parent):
    cdef NativeFrame result = NativeFrame()
    result.ptr = ptr
    result.frame_type = frame_type
    result.parent = parent
    parent.borrow_count += 1
    return result


cdef lf.NativeFrame *_copy_frame(lf.NativeFrame *source) except NULL:
    cdef lf.NativeFrame *result = new lf.NativeFrame(
        source.width, source.height, source.bytes_per_pixel, NULL, source.format
    )
    cdef size_t data_size = source.width * source.height * source.bytes_per_pixel
    if data_size:
        memcpy(result.data, source.data, data_size)
    result.timestamp = source.timestamp
    result.arrival_timestamp_us = source.arrival_timestamp_us
    result.sequence = source.sequence
    result.exposure = source.exposure
    result.gain = source.gain
    result.gamma = source.gamma
    result.status = source.status
    return result


cdef class NativeFrameSet:
    cdef map[lf.NativeFrameType, lf.NativeFrame *] frames
    cdef object listener
    cdef int borrow_count
    cdef bint filled
    cdef bint release_requested
    cdef bint released
    cdef bint owns_frames
    cdef bint aligned_listener
    cdef long long alignment_delta_ticks
    cdef long owner_pid

    def __cinit__(self):
        self.listener = None
        self.borrow_count = 0
        self.filled = False
        self.release_requested = False
        self.released = False
        self.owns_frames = False
        self.aligned_listener = False
        self.alignment_delta_ticks = -1
        self.owner_pid = c_getpid()

    def __dealloc__(self):
        if (self.owner_pid == c_getpid() and self.filled and
                not self.released):
            if self.listener is not None:
                self._release_via_listener()
            elif self.owns_frames:
                self._release_owned()

    cdef void _release_via_listener(self):
        if self.aligned_listener:
            (<NativeAlignedFrameListener>self.listener)._release_native(self)
        else:
            (<NativeSyncFrameListener>self.listener)._release_native(self)

    cdef void _release_owned(self):
        cdef map[lf.NativeFrameType, lf.NativeFrame *].iterator it = self.frames.begin()
        cdef lf.NativeFrame *frame
        while it != self.frames.end():
            frame = deref(it).second
            if frame != NULL:
                del frame
            inc(it)
        self.frames.clear()
        self.released = True

    cdef void _drop_borrow(self):
        if self.borrow_count > 0:
            self.borrow_count -= 1
        self._maybe_release()

    cdef void _maybe_release(self):
        if self.release_requested and self.borrow_count == 0 and not self.released:
            if self.listener is not None:
                self._release_via_listener()
            elif self.owns_frames:
                self._release_owned()
            else:
                self.released = True

    def get(self, int frame_type):
        _require_process(self.owner_pid)
        if self.release_requested:
            raise DeviceStateError("frame set release has already been requested")
        cdef lf.NativeFrameType key = _frame_type(frame_type)
        cdef map[lf.NativeFrameType, lf.NativeFrame *].iterator it = self.frames.find(key)
        if it == self.frames.end() or deref(it).second == NULL:
            raise KeyError(frame_type)
        return _borrow_frame(deref(it).second, frame_type, self)

    def contains(self, int frame_type):
        _require_process(self.owner_pid)
        if self.release_requested:
            return False
        cdef lf.NativeFrameType key = _frame_type(frame_type)
        cdef map[lf.NativeFrameType, lf.NativeFrame *].iterator it = self.frames.find(key)
        return it != self.frames.end() and deref(it).second != NULL

    def release(self):
        _require_process(self.owner_pid)
        if not self.release_requested:
            self.release_requested = True
        self._maybe_release()

    @property
    def is_released(self):
        _require_process(self.owner_pid)
        return self.release_requested != 0

    @property
    def release_complete(self):
        _require_process(self.owner_pid)
        return self.released != 0

    @property
    def entry_count(self):
        _require_process(self.owner_pid)
        return self.frames.size()

    @property
    def delta_ticks(self):
        _require_process(self.owner_pid)
        return None if self.alignment_delta_ticks < 0 else self.alignment_delta_ticks


def _testing_frame_set():
    """Build a tiny owned native set for hardware-free lifetime tests."""
    cdef NativeFrameSet result = NativeFrameSet()
    cdef lf.NativeFrame *color = new lf.NativeFrame(
        2, 1, 4, NULL, lf.FORMAT_BGRX
    )
    cdef lf.NativeFrame *depth = new lf.NativeFrame(
        2, 1, 4, NULL, lf.FORMAT_FLOAT
    )
    color.timestamp = 10
    color.arrival_timestamp_us = 1000
    color.sequence = 1
    depth.timestamp = 11
    depth.arrival_timestamp_us = 1100
    depth.sequence = 2
    result.frames[lf.FRAME_COLOR] = color
    result.frames[lf.FRAME_DEPTH] = depth
    result.filled = True
    result.owns_frames = True
    return result


cdef class NativeSyncFrameListener:
    cdef lf.NativeSyncListener *ptr
    cdef long owner_pid

    def __cinit__(self, unsigned int frame_types):
        self.owner_pid = c_getpid()
        self.ptr = new lf.NativeSyncListener(frame_types)

    def __dealloc__(self):
        if self.owner_pid == c_getpid() and self.ptr != NULL:
            del self.ptr
            self.ptr = NULL

    @property
    def _listener_address(self):
        _require_process(self.owner_pid)
        return <size_t><lf.NativeFrameListener *>self.ptr

    def has_new_frame(self):
        _require_process(self.owner_pid)
        return self.ptr.hasNewFrame() != 0

    def wait(self, timeout=None):
        _require_process(self.owner_pid)
        cdef NativeFrameSet result = NativeFrameSet()
        cdef bint ok = True
        cdef int milliseconds
        cdef double seconds
        result.listener = self
        if timeout is None:
            # The C++ wait cannot be interrupted once entered, so wait in
            # bounded slices and let Python signal handlers (Ctrl+C) run
            # between them.
            while True:
                with nogil:
                    ok = self.ptr.waitForNewFrame(result.frames, _WAIT_SLICE_MS)
                if ok:
                    break
                with nogil:
                    self.ptr.release(result.frames)
                PyErr_CheckSignals()
        else:
            seconds = float(timeout)
            if seconds < 0:
                raise ValueError("timeout must be non-negative")
            milliseconds = 0 if seconds == 0 else max(1, math.ceil(seconds * 1000.0))
            with nogil:
                ok = self.ptr.waitForNewFrame(result.frames, milliseconds)
        if not ok:
            result.filled = True
            self._release_native(result)
            raise FrameTimeoutError("timed out waiting for synchronized frames")
        result.filled = True
        return result

    cdef void _release_native(self, NativeFrameSet frame_set):
        _require_process(self.owner_pid)
        if self.ptr != NULL and frame_set.filled and not frame_set.released:
            with nogil:
                self.ptr.release(frame_set.frames)
            frame_set.released = True


cdef class NativeAlignedFrameListener:
    cdef lf.NativeAlignedListener *ptr
    cdef long owner_pid

    def __cinit__(self, unsigned int frame_types, uint32_t max_delta_ticks,
                  size_t queue_capacity=8):
        self.owner_pid = c_getpid()
        self.ptr = new lf.NativeAlignedListener(frame_types, max_delta_ticks, queue_capacity)

    def __dealloc__(self):
        if self.owner_pid == c_getpid() and self.ptr != NULL:
            del self.ptr
            self.ptr = NULL

    @property
    def _listener_address(self):
        _require_process(self.owner_pid)
        return <size_t><lf.NativeFrameListener *>self.ptr

    def has_new_frame(self):
        _require_process(self.owner_pid)
        return self.ptr.hasNewFrame() != 0

    def wait(self, timeout=None):
        _require_process(self.owner_pid)
        cdef NativeFrameSet result = NativeFrameSet()
        cdef lf.NativeAlignedListener.Statistics statistics
        cdef bint ok = True
        cdef int milliseconds
        cdef double seconds
        result.listener = self
        result.aligned_listener = True
        if timeout is None:
            while True:
                with nogil:
                    ok = self.ptr.waitForNewFrame(result.frames, _WAIT_SLICE_MS)
                if ok:
                    break
                with nogil:
                    self.ptr.release(result.frames)
                PyErr_CheckSignals()
        else:
            seconds = float(timeout)
            if seconds < 0:
                raise ValueError("timeout must be non-negative")
            milliseconds = 0 if seconds == 0 else max(1, math.ceil(seconds * 1000.0))
            with nogil:
                ok = self.ptr.waitForNewFrame(result.frames, milliseconds)
        if not ok:
            result.filled = True
            self._release_native(result)
            raise FrameTimeoutError("timed out waiting for timestamp-aligned frames")
        with nogil:
            statistics = self.ptr.getStatistics()
        result.alignment_delta_ticks = statistics.last_delta_ticks
        result.filled = True
        return result

    def statistics(self):
        _require_process(self.owner_pid)
        cdef lf.NativeAlignedListener.Statistics value
        with nogil:
            value = self.ptr.getStatistics()
        return {
            "delivered": value.delivered,
            "dropped": value.dropped,
            "last_delta_ticks": value.last_delta_ticks,
            "maximum_delta_ticks": value.maximum_delta_ticks,
        }

    def _testing_push(self, int frame_type, NativeFrame frame not None):
        """Inject a frame into the native queue for hardware-free tests."""
        _require_process(self.owner_pid)
        frame._check()
        cdef bint accepted
        cdef lf.NativeFrameType native_type = _frame_type(frame_type)
        cdef lf.NativeFrame *submitted = frame.ptr
        cdef lf.NativeFrame *copied = NULL
        if frame.numpy_owner is not None:
            copied = _copy_frame(frame.ptr)
            submitted = copied
        try:
            with nogil:
                accepted = self.ptr.onNewFrame(native_type, submitted)
        except Exception:
            if copied != NULL:
                del copied
            raise
        if accepted:
            if copied != NULL:
                del frame.ptr
                frame.numpy_owner = None
            frame.ptr = NULL
            frame.owns_ptr = False
        elif copied != NULL:
            del copied
        return accepted != 0

    cdef void _release_native(self, NativeFrameSet frame_set):
        _require_process(self.owner_pid)
        if self.ptr != NULL and frame_set.filled and not frame_set.released:
            with nogil:
                self.ptr.release(frame_set.frames)
            frame_set.released = True


cdef lf.NativeFrameListener *_listener_pointer(object listener) except NULL:
    if isinstance(listener, NativeSyncFrameListener):
        _require_process((<NativeSyncFrameListener>listener).owner_pid)
        return <lf.NativeFrameListener *>(<NativeSyncFrameListener>listener).ptr
    if isinstance(listener, NativeAlignedFrameListener):
        _require_process((<NativeAlignedFrameListener>listener).owner_pid)
        return <lf.NativeFrameListener *>(<NativeAlignedFrameListener>listener).ptr
    if isinstance(listener, NativeRecordingWriterHandle):
        _require_process((<NativeRecordingWriterHandle>listener).owner_pid)
        return <lf.NativeFrameListener *>(<NativeRecordingWriterHandle>listener).ptr
    raise TypeError("listener must be a native frame listener")


cdef str _depth_correction_name(lf.NativeDepthCorrectionModel value):
    if value == lf.DEPTH_CORRECTION_OFFSET:
        return "offset_only"
    if value == lf.DEPTH_CORRECTION_LINEAR:
        return "linear"
    raise ValueError(f"unknown native depth correction model: {<int>value}")


cdef dict _rigid_transform_mapping(lf.NativeRigidTransform value):
    cdef size_t index
    return {
        "rotation": tuple(lf.rigidRotation(value, index) for index in range(9)),
        "translation_m": tuple(lf.rigidTranslation(value, index) for index in range(3)),
    }


cdef dict _depth_correction_mapping(lf.NativeDepthCorrectionProfile value):
    return {
        "model": _depth_correction_name(value.model),
        "scale": value.scale,
        "offset_mm": value.offset_mm,
        "rmse_mm": value.rmse_mm,
    }


cdef dict _quality_mapping(lf.NativeCalibrationQualityMetrics value):
    return {
        "color_views": value.color_views,
        "ir_views": value.ir_views,
        "stereo_views": value.stereo_views,
        "depth_views": value.depth_views,
        "color_rms_px": value.color_rms_px,
        "ir_rms_px": value.ir_rms_px,
        "held_out_stereo_rms_px": value.held_out_stereo_rms_px,
        "depth_rmse_mm": value.depth_rmse_mm,
    }


cdef class NativeCalibrationProfileHandle:
    cdef lf.NativeCalibrationProfile *ptr
    cdef long owner_pid

    def __cinit__(self):
        self.owner_pid = c_getpid()
        self.ptr = new lf.NativeCalibrationProfile()

    def __dealloc__(self):
        if self.owner_pid == c_getpid() and self.ptr != NULL:
            del self.ptr
        self.ptr = NULL

    cdef void _check(self) except *:
        _require_process(self.owner_pid)
        if self.ptr == NULL:
            raise CalibrationError("calibration profile is no longer valid")

    @staticmethod
    def load(path):
        cdef NativeCalibrationProfileHandle result = NativeCalibrationProfileHandle()
        cdef string encoded = str(path).encode("utf-8")
        cdef string error
        cdef bint ok
        with nogil:
            ok = lf.NativeCalibrationProfile.load(encoded, deref(result.ptr), &error)
        if not ok:
            raise CalibrationError(_text(error))
        return result

    def save(self, path):
        self._check()
        cdef string encoded = str(path).encode("utf-8")
        cdef string error
        cdef bint ok
        with nogil:
            ok = self.ptr.save(encoded, &error)
        if not ok:
            raise CalibrationError(_text(error))

    def check_device(self, serial, firmware, bint allow_serial_mismatch=False):
        self._check()
        cdef string encoded_serial = str(serial).encode("utf-8")
        cdef string encoded_firmware = str(firmware).encode("utf-8")
        cdef string warning
        cdef string error
        cdef bint ok
        with nogil:
            ok = self.ptr.matchesDevice(
                encoded_serial, encoded_firmware, allow_serial_mismatch, &warning, &error
            )
        if not ok:
            raise CalibrationError(_text(error))
        return _text(warning) or None

    @property
    def schema_version(self):
        self._check()
        return self.ptr.schemaVersion()

    @property
    def serial(self):
        self._check()
        return _text(self.ptr.serial())

    @property
    def firmware(self):
        self._check()
        return _text(self.ptr.firmware())

    def color_camera(self):
        self._check()
        cdef lf.NativeProjectiveCameraModel value = self.ptr.colorCamera()
        return _projective_camera_mapping(value)

    def ir_camera(self):
        self._check()
        cdef lf.NativeProjectiveCameraModel value = self.ptr.irCamera()
        return _projective_camera_mapping(value)

    def depth_to_color(self):
        self._check()
        cdef lf.NativeRigidTransform value = self.ptr.depthToColor()
        return _rigid_transform_mapping(value)

    def depth_correction(self):
        self._check()
        if not self.ptr.hasDepthCorrection():
            return None
        cdef lf.NativeDepthCorrectionProfile value = self.ptr.depthCorrection()
        return _depth_correction_mapping(value)

    def quality_metrics(self):
        self._check()
        if not self.ptr.hasQualityMetrics():
            return None
        cdef lf.NativeCalibrationQualityMetrics value = self.ptr.qualityMetrics()
        return _quality_mapping(value)

    def provenance(self):
        self._check()
        return {
            "created_utc": _text(self.ptr.createdUtc()),
            "tool_version": _text(self.ptr.toolVersion()),
            "job_sha256": _text(self.ptr.jobSha256()),
        }


cdef int _rasterization(object value) except -1:
    cdef str name = str(value)
    if name == "nearest":
        return <int>lf.RASTERIZATION_NEAREST
    if name == "four_neighbor_splat":
        return <int>lf.RASTERIZATION_FOUR_NEIGHBOR
    raise ValueError(f"unknown registration rasterization: {value!r}")


cdef str _rasterization_name(lf.NativeRegistrationRasterization value):
    if value == lf.RASTERIZATION_NEAREST:
        return "nearest"
    if value == lf.RASTERIZATION_FOUR_NEIGHBOR:
        return "four_neighbor_splat"
    raise ValueError(f"unknown native registration rasterization: {<int>value}")


cdef dict _projective_options_mapping(lf.NativeProjectiveRegistrationOptions value):
    return {
        "rasterization": _rasterization_name(value.rasterization),
        "min_depth_mm": value.min_depth_mm,
        "max_depth_mm": value.max_depth_mm,
        "apply_depth_correction": value.apply_depth_correction != 0,
    }


cdef class NativeProjectiveRegistrationHandle:
    cdef lf.NativeProjectiveRegistration *ptr
    cdef long owner_pid

    def __cinit__(self, NativeCalibrationProfileHandle profile not None, target, options):
        profile._check()
        self.owner_pid = c_getpid()
        self.ptr = NULL
        cdef lf.NativeProjectiveCameraModel native_target
        cdef lf.NativeProjectiveRegistrationOptions native_options
        cdef string error
        cdef unique_ptr[lf.NativeProjectiveRegistration] created
        _fill_projective_camera(&native_target, target)
        native_options.rasterization = (
            <lf.NativeRegistrationRasterization>_rasterization(options.rasterization)
        )
        native_options.min_depth_mm = options.min_depth_mm
        native_options.max_depth_mm = options.max_depth_mm
        native_options.apply_depth_correction = options.apply_depth_correction
        with nogil:
            created = lf.NativeProjectiveRegistration.create(
                deref(profile.ptr), native_target, native_options, &error
            )
        if created.get() == NULL:
            raise CalibrationError(_text(error))
        self.ptr = created.release()

    def __dealloc__(self):
        if self.owner_pid == c_getpid() and self.ptr != NULL:
            del self.ptr
        self.ptr = NULL

    cdef void _check(self) except *:
        _require_process(self.owner_pid)
        if self.ptr == NULL:
            raise DeviceStateError("projective registration is no longer valid")

    def target_camera(self):
        self._check()
        cdef lf.NativeProjectiveCameraModel value = self.ptr.targetCamera()
        return _projective_camera_mapping(value)

    def options(self):
        self._check()
        cdef lf.NativeProjectiveRegistrationOptions value = self.ptr.options()
        return _projective_options_mapping(value)

    def apply(self, NativeFrame depth not None, NativeFrame output not None):
        self._check()
        depth._check()
        output._check()
        cdef string error
        cdef bint ok
        with nogil:
            ok = self.ptr.apply(deref(depth.ptr), deref(output.ptr), &error)
        if not ok:
            raise ValueError(_text(error))


cdef class NativeDeviceHandle:
    cdef lf.NativeDevice *ptr
    cdef object owner
    cdef object pipeline
    cdef object color_listener
    cdef object depth_listener
    cdef bint running
    cdef bint closed
    cdef long owner_pid

    def __cinit__(self):
        self.ptr = NULL
        self.owner = None
        self.pipeline = None
        self.color_listener = None
        self.depth_listener = None
        self.running = False
        self.closed = False
        self.owner_pid = c_getpid()

    def __dealloc__(self):
        cdef bint ignored
        if (self.owner_pid == c_getpid() and self.ptr != NULL and
                not self.closed):
            if self.running:
                with nogil:
                    ignored = self.ptr.stop()
            with nogil:
                self.ptr.setColorFrameListener(NULL)
                self.ptr.setIrAndDepthFrameListener(NULL)
                ignored = self.ptr.close()
        if self.owner_pid == c_getpid() and self.pipeline is not None:
            (<NativePipeline>self.pipeline)._device_closed()
        self.ptr = NULL

    cdef void _check(self) except *:
        _require_process(self.owner_pid)
        if self.ptr == NULL or self.closed:
            raise DeviceStateError("device is closed")

    cdef void _check_stopped(self) except *:
        self._check()
        if self.running:
            raise DeviceStateError("device configuration cannot change while streaming")

    @property
    def serial_number(self):
        self._check()
        return _text(self.ptr.getSerialNumber())

    @property
    def firmware_version(self):
        self._check()
        return _text(self.ptr.getFirmwareVersion())

    @property
    def pipeline_name(self):
        self._check()
        return _text(self.ptr.getPacketPipelineName())

    @property
    def state(self):
        _require_process(self.owner_pid)
        if self.ptr == NULL:
            return <int>lf.DEVICE_CLOSED
        cdef lf.NativeDeviceState value
        with nogil:
            value = self.ptr.getState()
        return <int>value

    @property
    def last_error(self):
        _require_process(self.owner_pid)
        if self.ptr == NULL:
            return ""
        cdef string value
        with nogil:
            value = self.ptr.getLastError()
        return _text(value)

    def runtime_statistics(self):
        self._check()
        cdef lf.NativeDeviceRuntimeStatistics value
        with nogil:
            value = self.ptr.getRuntimeStatistics()
        return {
            "color": {
                "decoded_frames": value.color.decoded_frames,
                "status_error_frames": value.color.status_error_frames,
                "sequence_gaps": value.color.sequence_gaps,
                "last_sequence": value.color.last_sequence,
                "last_device_timestamp": value.color.last_device_timestamp,
                "last_arrival_timestamp_us": value.color.last_arrival_timestamp_us,
            },
            "ir": {
                "decoded_frames": value.ir.decoded_frames,
                "status_error_frames": value.ir.status_error_frames,
                "sequence_gaps": value.ir.sequence_gaps,
                "last_sequence": value.ir.last_sequence,
                "last_device_timestamp": value.ir.last_device_timestamp,
                "last_arrival_timestamp_us": value.ir.last_arrival_timestamp_us,
            },
            "depth": {
                "decoded_frames": value.depth.decoded_frames,
                "status_error_frames": value.depth.status_error_frames,
                "sequence_gaps": value.depth.sequence_gaps,
                "last_sequence": value.depth.last_sequence,
                "last_device_timestamp": value.depth.last_device_timestamp,
                "last_arrival_timestamp_us": value.depth.last_arrival_timestamp_us,
            },
            "start_attempts": value.start_attempts,
            "successful_starts": value.successful_starts,
            "stop_calls": value.stop_calls,
            "disconnect_events": value.disconnect_events,
            "transfer_stall_events": value.transfer_stall_events,
        }

    def calibration_profile(self):
        self._check()
        cdef NativeCalibrationProfileHandle result = NativeCalibrationProfileHandle()
        cdef bint available
        with nogil:
            available = self.ptr.getCalibrationProfile(deref(result.ptr))
        return result if available else None

    def color_camera_params(self):
        self._check()
        cdef lf.NativeDevice.ColorCameraParams p = self.ptr.getColorCameraParams()
        return {
            "fx": p.fx, "fy": p.fy, "cx": p.cx, "cy": p.cy,
            "shift_d": p.shift_d, "shift_m": p.shift_m,
            "mx_x3y0": p.mx_x3y0, "mx_x0y3": p.mx_x0y3,
            "mx_x2y1": p.mx_x2y1, "mx_x1y2": p.mx_x1y2,
            "mx_x2y0": p.mx_x2y0, "mx_x0y2": p.mx_x0y2,
            "mx_x1y1": p.mx_x1y1, "mx_x1y0": p.mx_x1y0,
            "mx_x0y1": p.mx_x0y1, "mx_x0y0": p.mx_x0y0,
            "my_x3y0": p.my_x3y0, "my_x0y3": p.my_x0y3,
            "my_x2y1": p.my_x2y1, "my_x1y2": p.my_x1y2,
            "my_x2y0": p.my_x2y0, "my_x0y2": p.my_x0y2,
            "my_x1y1": p.my_x1y1, "my_x1y0": p.my_x1y0,
            "my_x0y1": p.my_x0y1, "my_x0y0": p.my_x0y0,
        }

    def ir_camera_params(self):
        self._check()
        cdef lf.NativeDevice.IrCameraParams p = self.ptr.getIrCameraParams()
        return {
            "fx": p.fx, "fy": p.fy, "cx": p.cx, "cy": p.cy,
            "k1": p.k1, "k2": p.k2, "k3": p.k3,
            "p1": p.p1, "p2": p.p2,
        }

    def set_color_camera_params(self, params):
        self._check_stopped()
        cdef lf.NativeDevice.ColorCameraParams value
        _fill_color(&value, params)
        with nogil:
            self.ptr.setColorCameraParams(value)

    def set_ir_camera_params(self, params):
        self._check_stopped()
        cdef lf.NativeDevice.IrCameraParams value
        _fill_ir(&value, params)
        with nogil:
            self.ptr.setIrCameraParams(value)

    def set_configuration(self, config):
        self._check_stopped()
        cdef lf.NativeDevice.Config value = lf.NativeDevice.Config()
        value.MinDepth = config.min_depth
        value.MaxDepth = config.max_depth
        value.EnableBilateralFilter = config.enable_bilateral_filter
        value.EnableEdgeAwareFilter = config.enable_edge_aware_filter
        with nogil:
            self.ptr.setConfiguration(value)

    def set_color_listener(self, listener not None):
        self._check_stopped()
        cdef lf.NativeFrameListener *listener_ptr = _listener_pointer(listener)
        with nogil:
            self.ptr.setColorFrameListener(listener_ptr)
        self.color_listener = listener

    def set_depth_listener(self, listener not None):
        self._check_stopped()
        cdef lf.NativeFrameListener *listener_ptr = _listener_pointer(listener)
        with nogil:
            self.ptr.setIrAndDepthFrameListener(listener_ptr)
        self.depth_listener = listener

    def set_color_auto_exposure(self, float compensation=0.0):
        self._check_stopped()
        if compensation < -2.0 or compensation > 2.0:
            raise ValueError("exposure compensation must be in [-2, 2]")
        with nogil:
            self.ptr.setColorAutoExposure(compensation)

    def set_color_semi_auto_exposure(self, float exposure_time_ms):
        self._check_stopped()
        if exposure_time_ms <= 0:
            raise ValueError("exposure time must be positive")
        with nogil:
            self.ptr.setColorSemiAutoExposure(exposure_time_ms)

    def set_color_manual_exposure(self, float integration_time_ms, float analog_gain):
        self._check_stopped()
        if integration_time_ms <= 0 or integration_time_ms > 66:
            raise ValueError("integration_time_ms must be in (0, 66]")
        if analog_gain < 1 or analog_gain > 4:
            raise ValueError("analog_gain must be in [1, 4]")
        with nogil:
            self.ptr.setColorManualExposure(integration_time_ms, analog_gain)

    def set_color_setting(self, int command, value):
        self._check_stopped()
        cdef lf.NativeColorSetting native_command = <lf.NativeColorSetting>command
        cdef float float_value
        cdef uint32_t int_value
        if isinstance(value, (float, np.floating)):
            float_value = float(value)
            with nogil:
                self.ptr.setColorSetting(native_command, float_value)
        # Booleans are integers in Python, but they are not valid color
        # setting values, so reject them explicitly.
        elif (isinstance(value, (int, np.integer)) and
              not isinstance(value, (bool, np.bool_))):
            if not 0 <= int(value) <= 0xFFFFFFFF:
                raise ValueError("integer color settings must fit in uint32")
            int_value = int(value)
            with nogil:
                self.ptr.setColorSetting(native_command, int_value)
        else:
            raise TypeError("color settings must be an integer or floating-point value")

    def get_color_setting(self, int command, bint as_float=False):
        self._check()
        cdef lf.NativeColorSetting native_command = <lf.NativeColorSetting>command
        cdef float float_value
        cdef uint32_t int_value
        if as_float:
            with nogil:
                float_value = self.ptr.getColorSettingFloat(native_command)
            return float_value
        with nogil:
            int_value = self.ptr.getColorSetting(native_command)
        return int_value

    def set_led_status(self, settings):
        self._check_stopped()
        cdef lf.NativeLedSettings value
        value.LedId = settings.led_id
        value.Mode = settings.mode
        value.StartLevel = settings.start_level
        value.StopLevel = settings.stop_level
        value.IntervalInMs = settings.interval_ms
        value.Reserved = settings.reserved
        with nogil:
            self.ptr.setLedStatus(value)

    def start(self, rgb=True, depth=True):
        self._check()
        if self.running:
            raise DeviceStateError("device is already streaming")
        cdef bint ok
        cdef bint enable_rgb = rgb
        cdef bint enable_depth = depth
        if not enable_rgb and not enable_depth:
            raise ValueError("at least one stream must be enabled")
        if enable_rgb and self.color_listener is None:
            raise DeviceStateError("a color listener must be attached before starting RGB")
        if enable_depth and self.depth_listener is None:
            raise DeviceStateError("a depth listener must be attached before starting IR/depth")
        with nogil:
            ok = self.ptr.startStreams(enable_rgb, enable_depth)
        if not ok:
            raise DeviceStateError("device failed to start the requested streams")
        self.running = True

    def stop(self):
        _require_process(self.owner_pid)
        if self.ptr == NULL or self.closed:
            return
        cdef bint ok = True
        if self.running:
            with nogil:
                ok = self.ptr.stop()
        self.running = False
        if not ok:
            raise DeviceStateError("device failed to stop")

    def close(self):
        _require_process(self.owner_pid)
        if self.closed or self.ptr == NULL:
            return
        cdef bint stop_ok = True
        cdef bint close_ok
        if self.running:
            with nogil:
                stop_ok = self.ptr.stop()
        self.running = False
        with nogil:
            self.ptr.setColorFrameListener(NULL)
            self.ptr.setIrAndDepthFrameListener(NULL)
            close_ok = self.ptr.close()
        self.color_listener = None
        self.depth_listener = None
        self.closed = True
        if self.pipeline is not None:
            (<NativePipeline>self.pipeline)._device_closed()
            self.pipeline = None
        if not stop_ok:
            raise DeviceStateError("device failed to stop while closing")
        if not close_ok:
            raise DeviceStateError("device failed to close")

    @property
    def is_running(self):
        _require_process(self.owner_pid)
        return self.running != 0

    @property
    def is_closed(self):
        _require_process(self.owner_pid)
        return self.closed != 0


cdef NativeDeviceHandle _wrap_device(lf.NativeDevice *ptr, object owner, NativePipeline pipeline):
    if ptr == NULL:
        if pipeline is not None:
            pipeline._open_failed()
        raise DeviceOpenError("libfreenect2 returned no device")
    cdef NativeDeviceHandle result = NativeDeviceHandle()
    result.ptr = ptr
    result.owner = owner
    result.pipeline = pipeline
    if pipeline is not None:
        pipeline._attach()
    return result


cdef class NativeRecordingWriterHandle:
    cdef lf.NativeRecordingWriter *ptr
    cdef long owner_pid

    def __cinit__(self, path, size_t queue_capacity=32):
        cdef string encoded = str(path).encode("utf-8")
        cdef string error
        self.owner_pid = c_getpid()
        self.ptr = NULL
        self.ptr = new lf.NativeRecordingWriter(encoded, queue_capacity)
        if not self.ptr.isOpen():
            error = self.ptr.getLastError()
            del self.ptr
            self.ptr = NULL
            raise RecordingError(_text(error))

    def __dealloc__(self):
        if self.owner_pid == c_getpid() and self.ptr != NULL:
            del self.ptr
        self.ptr = NULL

    cdef void _check(self) except *:
        _require_process(self.owner_pid)
        if self.ptr == NULL:
            raise RecordingError("recording writer is no longer valid")

    @property
    def _listener_address(self):
        self._check()
        return <size_t><lf.NativeFrameListener *>self.ptr

    @property
    def is_open(self):
        self._check()
        cdef bint value
        with nogil:
            value = self.ptr.isOpen()
        return value != 0

    @property
    def last_error(self):
        self._check()
        cdef string value
        with nogil:
            value = self.ptr.getLastError()
        return _text(value)

    def publish_calibration(self, NativeDeviceHandle device not None):
        self._check()
        device._check()
        cdef lf.NativeCalibrationData calibration
        cdef string serial = device.ptr.getSerialNumber()
        cdef string firmware = device.ptr.getFirmwareVersion()
        cdef bint ok
        with nogil:
            ok = device.ptr.getCalibrationData(calibration)
        if not ok:
            raise RecordingError("device calibration is not available")
        with nogil:
            ok = self.ptr.setCalibration(serial, firmware, calibration)
        if not ok:
            raise RecordingError(self.last_error)

    def set_calibration_profile(
        self, NativeCalibrationProfileHandle profile not None,
        bint allow_serial_mismatch=False
    ):
        self._check()
        profile._check()
        cdef bint ok
        with nogil:
            ok = self.ptr.setCalibrationProfile(
                deref(profile.ptr), allow_serial_mismatch
            )
        if not ok:
            raise CalibrationError(self.last_error)

    def statistics(self):
        self._check()
        cdef lf.NativeRecordingWriter.Stats value
        with nogil:
            value = self.ptr.getStats()
        return {
            "written_frames": value.written_frames,
            "written_color_frames": value.written_color_frames,
            "written_depth_frames": value.written_depth_frames,
            "dropped_frames": value.dropped_frames,
            "written_bytes": value.written_bytes,
        }

    def close(self):
        self._check()
        cdef bint ok
        with nogil:
            ok = self.ptr.close()
        if not ok:
            raise RecordingError(self.last_error)


cdef class NativeFreenect2Context:
    cdef lf.NativeFreenect2 *ptr
    cdef long owner_pid

    def __cinit__(self):
        self.owner_pid = c_getpid()
        self.ptr = new lf.NativeFreenect2()

    def __dealloc__(self):
        if self.owner_pid == c_getpid() and self.ptr != NULL:
            del self.ptr
            self.ptr = NULL

    def enumerate_devices(self):
        _require_process(self.owner_pid)
        cdef int count
        with nogil:
            count = self.ptr.enumerateDevices()
        return count

    def device_serial_number(self, int index):
        _require_process(self.owner_pid)
        return _text(self.ptr.getDeviceSerialNumber(index))

    def default_device_serial_number(self):
        _require_process(self.owner_pid)
        return _text(self.ptr.getDefaultDeviceSerialNumber())

    def wait_for_device(self, serial, double timeout, double poll_interval=0.25):
        _require_process(self.owner_pid)
        if timeout < 0 or not math.isfinite(timeout):
            raise ValueError("timeout must be finite and non-negative")
        if poll_interval <= 0 or not math.isfinite(poll_interval):
            raise ValueError("poll_interval must be finite and positive")
        cdef string encoded = str(serial).encode("utf-8")
        if encoded.empty():
            raise ValueError("serial must not be empty")
        cdef double deadline = time.monotonic() + timeout
        cdef double remaining
        cdef uint32_t slice_ms
        cdef uint32_t poll_ms = max(1, min(0xFFFFFFFF, math.ceil(poll_interval * 1000.0)))
        cdef bint found
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                # Preserve the core's single enumeration attempt for a zero timeout.
                slice_ms = 0
            else:
                slice_ms = max(1, min(100, math.ceil(remaining * 1000.0)))
            with nogil:
                found = self.ptr.waitForDevice(encoded, slice_ms, poll_ms)
            if found:
                return True
            PyErr_CheckSignals()
            if remaining <= 0 or time.monotonic() >= deadline:
                return False

    def open_device(self, name=None, NativePipeline pipeline=None):
        _require_process(self.owner_pid)
        cdef lf.NativeDevice *device = NULL
        cdef lf.NativePacketPipeline *native_pipeline = NULL
        cdef string serial
        cdef int index
        if pipeline is not None:
            native_pipeline = pipeline._consume()
        if name is None:
            with nogil:
                if native_pipeline == NULL:
                    device = self.ptr.openDefaultDevice()
                else:
                    device = self.ptr.openDefaultDevice(native_pipeline)
        elif isinstance(name, int):
            index = name
            with nogil:
                if native_pipeline == NULL:
                    device = self.ptr.openDevice(index)
                else:
                    device = self.ptr.openDevice(index, native_pipeline)
        else:
            serial = str(name).encode("utf-8")
            with nogil:
                if native_pipeline == NULL:
                    device = self.ptr.openDevice(serial)
                else:
                    device = self.ptr.openDevice(serial, native_pipeline)
        return _wrap_device(device, self, pipeline)


cdef class NativeReplayContext:
    cdef lf.NativeReplay *ptr
    cdef long owner_pid

    def __cinit__(self):
        self.owner_pid = c_getpid()
        self.ptr = new lf.NativeReplay()

    def __dealloc__(self):
        if self.owner_pid == c_getpid() and self.ptr != NULL:
            del self.ptr
            self.ptr = NULL

    def open_device(self, filenames, calibration=None, NativePipeline pipeline=None):
        _require_process(self.owner_pid)
        cdef vector[string] paths
        cdef object filename
        for filename in filenames:
            paths.push_back(str(filename).encode("utf-8"))
        if paths.empty():
            raise DeviceOpenError("replay requires at least one frame filename")

        cdef lf.NativePacketPipeline *native_pipeline = NULL
        if pipeline is not None:
            native_pipeline = pipeline._consume()
        cdef lf.NativeDevice *device = NULL
        cdef lf.ReplayCalibration native_calibration
        cdef cnp.ndarray p0
        cdef cnp.ndarray x_table
        cdef cnp.ndarray z_table
        cdef cnp.ndarray lookup_table
        if calibration is None:
            with nogil:
                if native_pipeline == NULL:
                    device = self.ptr.openDevice(paths)
                else:
                    device = self.ptr.openDevice(paths, native_pipeline)
        else:
            _fill_color(&native_calibration.color, calibration.color)
            _fill_ir(&native_calibration.ir, calibration.ir)
            p0 = np.ascontiguousarray(calibration.p0_tables, dtype=np.uint8)
            x_table = np.ascontiguousarray(calibration.x_table, dtype=np.float32)
            z_table = np.ascontiguousarray(calibration.z_table, dtype=np.float32)
            lookup_table = np.ascontiguousarray(calibration.lookup_table, dtype=np.int16)
            native_calibration.p0_tables.resize(p0.size)
            native_calibration.x_table.resize(x_table.size)
            native_calibration.z_table.resize(z_table.size)
            native_calibration.lookup_table.resize(lookup_table.size)
            if p0.size:
                memcpy(&native_calibration.p0_tables[0], cnp.PyArray_DATA(p0), p0.nbytes)
            if x_table.size:
                memcpy(&native_calibration.x_table[0], cnp.PyArray_DATA(x_table), x_table.nbytes)
            if z_table.size:
                memcpy(&native_calibration.z_table[0], cnp.PyArray_DATA(z_table), z_table.nbytes)
            if lookup_table.size:
                memcpy(&native_calibration.lookup_table[0], cnp.PyArray_DATA(lookup_table), lookup_table.nbytes)
            with nogil:
                if native_pipeline == NULL:
                    device = self.ptr.openDevice(paths, native_calibration)
                else:
                    device = self.ptr.openDevice(paths, native_calibration, native_pipeline)
        return _wrap_device(device, self, pipeline)

    def open_recording(self, path, options, NativePipeline pipeline=None):
        _require_process(self.owner_pid)
        cdef string directory = str(path).encode("utf-8")
        if directory.empty():
            raise DeviceOpenError("recording path must not be empty")
        cdef lf.NativeReplayOptions native_options = lf.NativeReplayOptions()
        native_options.salvage_incomplete = options.salvage_incomplete
        native_options.reproduce_timing = options.reproduce_timing
        cdef lf.NativePacketPipeline *native_pipeline = NULL
        cdef lf.NativeDevice *device = NULL
        if pipeline is not None:
            native_pipeline = pipeline._consume()
        with nogil:
            if native_pipeline == NULL:
                device = self.ptr.openRecording(directory, native_options)
            else:
                device = self.ptr.openRecording(
                    directory, native_pipeline, native_options
                )
        return _wrap_device(device, self, pipeline)


cdef class NativeRegistrationHandle:
    # Deliberately not fork-guarded: registration is pure CPU math over
    # tables copied at construction, so unlike device, listener, and frame
    # state it is safe to use and free in a forked child.
    cdef lf.NativeRegistration *ptr

    def __cinit__(self, ir_params, color_params):
        cdef lf.NativeDevice.IrCameraParams ir
        cdef lf.NativeDevice.ColorCameraParams color
        _fill_ir(&ir, ir_params)
        _fill_color(&color, color_params)
        self.ptr = new lf.NativeRegistration(ir, color)

    def __dealloc__(self):
        if self.ptr != NULL:
            del self.ptr
            self.ptr = NULL

    cdef void _check(self) except *:
        if self.ptr == NULL:
            raise DeviceStateError("registration handle is closed")

    def apply_point(self, int dx, int dy, float dz):
        self._check()
        cdef float cx = 0
        cdef float cy = 0
        with nogil:
            self.ptr.apply(dx, dy, dz, cx, cy)
        return cx, cy

    def apply(self, NativeFrame rgb not None, NativeFrame depth not None,
              NativeFrame undistorted not None, NativeFrame registered not None,
              bint enable_filter=True,
              NativeFrame bigdepth=None, object color_depth_map=None):
        self._check()
        rgb._check()
        depth._check()
        undistorted._check()
        registered._check()
        cdef lf.NativeFrame *big = NULL
        if bigdepth is not None:
            bigdepth._check()
            big = bigdepth.ptr
        cdef cnp.ndarray mapping
        cdef int *mapping_ptr = NULL
        if color_depth_map is not None:
            mapping = np.asarray(color_depth_map)
            if (mapping.dtype != np.dtype(np.int32) or
                    not mapping.flags.c_contiguous or mapping.size != 512 * 424):
                raise ValueError("color_depth_map must be a contiguous int32 array with 512*424 entries")
            mapping_ptr = <int *>cnp.PyArray_DATA(mapping)
        with nogil:
            self.ptr.apply(rgb.ptr, depth.ptr, undistorted.ptr, registered.ptr,
                           enable_filter, big, mapping_ptr)

    def undistort_depth(self, NativeFrame depth not None,
                        NativeFrame undistorted not None):
        self._check()
        depth._check()
        undistorted._check()
        with nogil:
            self.ptr.undistortDepth(depth.ptr, undistorted.ptr)

    def point_xyz(self, NativeFrame undistorted not None, int row, int column):
        self._check()
        undistorted._check()
        cdef float x = 0
        cdef float y = 0
        cdef float z = 0
        with nogil:
            self.ptr.getPointXYZ(undistorted.ptr, row, column, x, y, z)
        return x, y, z

    def points_xyz(self, NativeFrame undistorted not None, object pixels):
        self._check()
        undistorted._check()
        cdef cnp.ndarray coordinates = np.asarray(pixels)
        if (coordinates.dtype != np.dtype(np.int32) or coordinates.ndim != 2 or
                coordinates.shape[1] != 2 or not coordinates.flags.c_contiguous):
            raise ValueError("pixels must be a contiguous (N, 2) int32 array")
        cdef cnp.ndarray output = np.empty((coordinates.shape[0], 3), dtype=np.float32)
        cdef int32_t *pixel_data = <int32_t *>cnp.PyArray_DATA(coordinates)
        cdef float *xyz = <float *>cnp.PyArray_DATA(output)
        cdef size_t count = coordinates.shape[0]
        cdef size_t i
        with nogil:
            for i in range(count):
                self.ptr.getPointXYZ(
                    undistorted.ptr, pixel_data[i * 2], pixel_data[i * 2 + 1],
                    xyz[i * 3], xyz[i * 3 + 1], xyz[i * 3 + 2]
                )
        return output

    def build_color_to_depth_map(self, NativeFrame undistorted not None,
                                 object depth_to_color, object color_to_depth):
        self._check()
        undistorted._check()
        cdef cnp.ndarray forward = np.asarray(depth_to_color)
        cdef cnp.ndarray reverse = np.asarray(color_to_depth)
        if (forward.dtype != np.dtype(np.int32) or forward.size != 512 * 424 or
                not forward.flags.c_contiguous):
            raise ValueError("depth_to_color_map must be contiguous int32 with shape (424, 512)")
        if (reverse.dtype != np.dtype(np.int32) or reverse.size != 1920 * 1080 or
                not reverse.flags.c_contiguous or not reverse.flags.writeable):
            raise ValueError("color_to_depth_map must be writable contiguous int32 with shape (1080, 1920)")
        cdef bint ok
        cdef int *forward_ptr = <int *>cnp.PyArray_DATA(forward)
        cdef int32_t *reverse_ptr = <int32_t *>cnp.PyArray_DATA(reverse)
        cdef size_t forward_count = forward.size
        cdef size_t reverse_count = reverse.size
        with nogil:
            ok = lf.buildColorToDepthMap(
                undistorted.ptr, forward_ptr, forward_count, reverse_ptr, reverse_count
            )
        if not ok:
            raise ValueError("could not build color-to-depth map")

    def lift_normalized(self, NativeFrame undistorted not None, object color_to_depth,
                        object normalized_xy, int primary_radius, int fallback_radius,
                        float cluster_span_mm):
        self._check()
        undistorted._check()
        cdef cnp.ndarray reverse = np.asarray(color_to_depth)
        cdef cnp.ndarray points = np.asarray(normalized_xy)
        if (reverse.dtype != np.dtype(np.int32) or reverse.ndim != 2 or
                reverse.shape[0] != 1080 or reverse.shape[1] != 1920 or
                not reverse.flags.c_contiguous):
            raise ValueError("color_to_depth_map must be contiguous (1080, 1920) int32")
        if (points.dtype != np.dtype(np.float32) or points.ndim != 2 or
                points.shape[1] != 2 or not points.flags.c_contiguous):
            raise ValueError("normalized points must be contiguous (N, 2) float32")
        cdef cnp.ndarray xyz = np.empty((points.shape[0], 3), dtype=np.float32)
        cdef cnp.ndarray valid = np.empty(points.shape[0], dtype=np.bool_)
        cdef cnp.ndarray indices = np.empty(points.shape[0], dtype=np.int32)
        cdef lf.NativeDepthSearchOptions options = lf.NativeDepthSearchOptions()
        options.primary_radius = primary_radius
        options.fallback_radius = fallback_radius
        options.cluster_span_mm = cluster_span_mm
        cdef bint ok
        cdef int32_t *reverse_ptr = <int32_t *>cnp.PyArray_DATA(reverse)
        cdef float *points_ptr = <float *>cnp.PyArray_DATA(points)
        cdef float *xyz_ptr = <float *>cnp.PyArray_DATA(xyz)
        cdef uint8_t *valid_ptr = <uint8_t *>cnp.PyArray_DATA(valid)
        cdef int32_t *indices_ptr = <int32_t *>cnp.PyArray_DATA(indices)
        cdef size_t reverse_count = reverse.size
        cdef size_t point_count = points.shape[0]
        with nogil:
            ok = lf.liftColorPoints(
                self.ptr, undistorted.ptr, reverse_ptr, reverse_count, 1920, 1080,
                points_ptr, point_count, options, xyz_ptr, valid_ptr, indices_ptr
            )
        if not ok:
            raise ValueError("could not lift normalized color points")
        return xyz, valid, indices

    def point_xyz_rgb(self, NativeFrame undistorted not None,
                      NativeFrame registered not None, int row, int column):
        self._check()
        undistorted._check()
        registered._check()
        cdef float x = 0
        cdef float y = 0
        cdef float z = 0
        cdef float rgb = 0
        with nogil:
            self.ptr.getPointXYZRGB(undistorted.ptr, registered.ptr, row, column, x, y, z, rgb)
        cdef unsigned char *channels = <unsigned char *>&rgb
        if registered.ptr.format == lf.FORMAT_BGRX:
            return x, y, z, channels[2], channels[1], channels[0]
        return x, y, z, channels[0], channels[1], channels[2]


def default_logger_level():
    return <int>lf.loggerDefaultLevel()


def logger_level_name(int level):
    return _text(lf.loggerLevelToString(<lf.NativeLoggerLevel>level))


def global_logger_level():
    cdef lf.NativeLogger *logger = lf.getGlobalLogger()
    if logger == NULL:
        return None
    return <int>logger.level()


def set_global_log_level(level=None):
    cdef lf.NativeLogger *logger = NULL
    if level is not None:
        logger = lf.createConsoleLogger(<lf.NativeLoggerLevel><int>level)
    lf.setGlobalLogger(logger)
