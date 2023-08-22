import pytest
from pytest import raises

from ggml import lib, ffi
from ggml.utils import init, copy, numpy
import numpy as np
import numpy.testing as npt

@pytest.fixture()
def ctx():
    print("setup")
    yield init(mem_size=10*1024*1024)
    print("teardown")

class TestNumPy:
    
    # Single element

    def test_set_get_single_i32(self, ctx):
        i = lib.ggml_new_i32(ctx, 42)
        assert lib.ggml_get_i32_1d(i, 0) == 42
        assert numpy(i) == np.array([42], dtype=np.int32)

    def test_set_get_single_f32(self, ctx):
        i = lib.ggml_new_f32(ctx, 4.2)
        
        epsilon = 0.000001 # Not sure why so large a difference??
        pytest.approx(lib.ggml_get_f32_1d(i, 0), 4.2, epsilon)
        pytest.approx(numpy(i), np.array([4.2], dtype=np.float32), epsilon)

    def _test_copy_np_to_ggml(self, a: np.ndarray, t: ffi.CData):
        a2 = a.copy() # Clone original
        copy(a, t)
        npt.assert_array_equal(numpy(t), a2)

    # I32

    def test_copy_np_to_ggml_1d_i32(self, ctx):
        t = lib.ggml_new_tensor_1d(ctx, lib.GGML_TYPE_I32, 10)
        a = np.arange(10, dtype=np.int32)
        self._test_copy_np_to_ggml(a, t)

    def test_copy_np_to_ggml_2d_i32(self, ctx):
        t = lib.ggml_new_tensor_2d(ctx, lib.GGML_TYPE_I32, 2, 3)
        a = np.arange(2 * 3, dtype=np.int32).reshape((2, 3))
        self._test_copy_np_to_ggml(a, t)

    def test_copy_np_to_ggml_3d_i32(self, ctx):
        t = lib.ggml_new_tensor_3d(ctx, lib.GGML_TYPE_I32, 2, 3, 4)
        a = np.arange(2 * 3 * 4, dtype=np.int32).reshape((2, 3, 4))
        self._test_copy_np_to_ggml(a, t)

    def test_copy_np_to_ggml_4d_i32(self, ctx):
        t = lib.ggml_new_tensor_4d(ctx, lib.GGML_TYPE_I32, 2, 3, 4, 5)
        a = np.arange(2 * 3 * 4 * 5, dtype=np.int32).reshape((2, 3, 4, 5))
        self._test_copy_np_to_ggml(a, t)

    def test_copy_np_to_ggml_4d_n_i32(self, ctx):
        dims = [2, 3, 4, 5] # GGML_MAX_DIMS is 4, going beyond would crash
        pdims = ffi.new('int64_t[]', len(dims))
        for i, d in enumerate(dims): pdims[i] = d
        t = lib.ggml_new_tensor(ctx, lib.GGML_TYPE_I32, len(dims), pdims)
        a = np.arange(np.prod(dims), dtype=np.int32).reshape(tuple(pdims))
        self._test_copy_np_to_ggml(a, t)

    # F32

    def test_copy_np_to_ggml_1d_f32(self, ctx):
        t = lib.ggml_new_tensor_1d(ctx, lib.GGML_TYPE_F32, 10)
        a = np.arange(10, dtype=np.float32)
        self._test_copy_np_to_ggml(a, t)

    def test_copy_np_to_ggml_2d_f32(self, ctx):
        t = lib.ggml_new_tensor_2d(ctx, lib.GGML_TYPE_F32, 2, 3)
        a = np.arange(2 * 3, dtype=np.float32).reshape((2, 3))
        self._test_copy_np_to_ggml(a, t)

    def test_copy_np_to_ggml_3d_f32(self, ctx):
        t = lib.ggml_new_tensor_3d(ctx, lib.GGML_TYPE_F32, 2, 3, 4)
        a = np.arange(2 * 3 * 4, dtype=np.float32).reshape((2, 3, 4))
        self._test_copy_np_to_ggml(a, t)

    def test_copy_np_to_ggml_4d_f32(self, ctx):
        t = lib.ggml_new_tensor_4d(ctx, lib.GGML_TYPE_F32, 2, 3, 4, 5)
        a = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape((2, 3, 4, 5))
        self._test_copy_np_to_ggml(a, t)

    def test_copy_np_to_ggml_4d_n_f32(self, ctx):
        dims = [2, 3, 4, 5] # GGML_MAX_DIMS is 4, going beyond would crash
        pdims = ffi.new('int64_t[]', len(dims))
        for i, d in enumerate(dims): pdims[i] = d
        t = lib.ggml_new_tensor(ctx, lib.GGML_TYPE_F32, len(dims), pdims)
        a = np.arange(np.prod(dims), dtype=np.float32).reshape(tuple(pdims))
        self._test_copy_np_to_ggml(a, t)

    # F16

    def test_copy_np_to_ggml_1d_f16(self, ctx):
        t = lib.ggml_new_tensor_1d(ctx, lib.GGML_TYPE_F16, 10)
        a = np.arange(10, dtype=np.float16)
        self._test_copy_np_to_ggml(a, t)

    def test_copy_np_to_ggml_2d_f16(self, ctx):
        t = lib.ggml_new_tensor_2d(ctx, lib.GGML_TYPE_F16, 2, 3)
        a = np.arange(2 * 3, dtype=np.float16).reshape((2, 3))
        self._test_copy_np_to_ggml(a, t)

    def test_copy_np_to_ggml_3d_f16(self, ctx):
        t = lib.ggml_new_tensor_3d(ctx, lib.GGML_TYPE_F16, 2, 3, 4)
        a = np.arange(2 * 3 * 4, dtype=np.float16).reshape((2, 3, 4))
        self._test_copy_np_to_ggml(a, t)

    def test_copy_np_to_ggml_4d_f16(self, ctx):
        t = lib.ggml_new_tensor_4d(ctx, lib.GGML_TYPE_F16, 2, 3, 4, 5)
        a = np.arange(2 * 3 * 4 * 5, dtype=np.float16).reshape((2, 3, 4, 5))
        self._test_copy_np_to_ggml(a, t)

    def test_copy_np_to_ggml_4d_n_f16(self, ctx):
        dims = [2, 3, 4, 5] # GGML_MAX_DIMS is 4, going beyond would crash
        pdims = ffi.new('int64_t[]', len(dims))
        for i, d in enumerate(dims): pdims[i] = d
        t = lib.ggml_new_tensor(ctx, lib.GGML_TYPE_F16, len(dims), pdims)
        a = np.arange(np.prod(dims), dtype=np.float16).reshape(tuple(pdims))
        self._test_copy_np_to_ggml(a, t)

    # Mismatching shapes

    def test_copy_mismatching_shapes_1d(self, ctx):
        t = lib.ggml_new_tensor_1d(ctx, lib.GGML_TYPE_F32, 10)
        a = np.arange(10, dtype=np.float32)
        copy(a, t) # OK
        
        a = a.reshape((5, 2))
        with raises(AssertionError): copy(a, t)
        with raises(AssertionError): copy(t, a)
            
    def test_copy_mismatching_shapes_2d(self, ctx):
        t = lib.ggml_new_tensor_2d(ctx, lib.GGML_TYPE_F32, 2, 3)
        a = np.arange(6, dtype=np.float32)
        copy(a.reshape((2, 3)), t) # OK
        
        a = a.reshape((3, 2))
        with raises(AssertionError): copy(a, t)
        with raises(AssertionError): copy(t, a)

    def test_copy_mismatching_shapes_3d(self, ctx):
        t = lib.ggml_new_tensor_3d(ctx, lib.GGML_TYPE_F32, 2, 3, 4)
        a = np.arange(24, dtype=np.float32)
        copy(a.reshape((2, 3, 4)), t) # OK
        
        a = a.reshape((2, 4, 3))
        with raises(AssertionError): copy(a, t)
        with raises(AssertionError): copy(t, a)

    def test_copy_mismatching_shapes_4d(self, ctx):
        t = lib.ggml_new_tensor_4d(ctx, lib.GGML_TYPE_F32, 2, 3, 4, 5)
        a = np.arange(24*5, dtype=np.float32)
        copy(a.reshape((2, 3, 4, 5)), t) # OK
        
        a = a.reshape((2, 3, 5, 4))
        with raises(AssertionError): copy(a, t)
        with raises(AssertionError): copy(t, a)

    def test_copy_f16_to_f32(self, ctx):
        t = lib.ggml_new_tensor_1d(ctx, lib.GGML_TYPE_F32, 1)
        a = np.array([123.45], dtype=np.float16)
        copy(a, t)
        np.testing.assert_allclose(lib.ggml_get_f32_1d(t, 0), 123.45, rtol=1e-3)

    def test_copy_f32_to_f16(self, ctx):
        t = lib.ggml_new_tensor_1d(ctx, lib.GGML_TYPE_F16, 1)
        a = np.array([123.45], dtype=np.float32)
        copy(a, t)
        np.testing.assert_allclose(lib.ggml_get_f32_1d(t, 0), 123.45, rtol=1e-3)

    def test_copy_f16_to_Q5_K(self, ctx):
        n = 256
        t = lib.ggml_new_tensor_1d(ctx, lib.GGML_TYPE_Q5_K, n)
        a = np.arange(n, dtype=np.float16)
        copy(a, t)
        np.testing.assert_allclose(a, numpy(t, allow_copy=True), rtol=0.05)

    def test_copy_Q5_K_to_f16(self, ctx):
        n = 256
        t = lib.ggml_new_tensor_1d(ctx, lib.GGML_TYPE_Q5_K, n)
        copy(np.arange(n, dtype=np.float32), t)
        a = np.arange(n, dtype=np.float16)
        copy(t, a)
        np.testing.assert_allclose(a, numpy(t, allow_copy=True), rtol=0.05)

    def test_copy_i16_f32_mismatching_types(self, ctx):
        t = lib.ggml_new_tensor_1d(ctx, lib.GGML_TYPE_F32, 1)
        a = np.arange(1, dtype=np.int16)
        with raises(NotImplementedError): copy(a, t)
        with raises(NotImplementedError): copy(t, a)

class TestTensorCopy:

    def test_copy_self(self, ctx):
        t = lib.ggml_new_i32(ctx, 42)
        copy(t, t)
        assert lib.ggml_get_i32_1d(t, 0) == 42

    def test_copy_1d(self, ctx):
        t1 = lib.ggml_new_tensor_1d(ctx, lib.GGML_TYPE_F32, 10)
        t2 = lib.ggml_new_tensor_1d(ctx, lib.GGML_TYPE_F32, 10)
        a = np.arange(10, dtype=np.float32)
        copy(a, t1)
        copy(t1, t2)
        assert np.allclose(a, numpy(t2))
        assert np.allclose(numpy(t1), numpy(t2))

class TestGraph:

    def test_add(self, ctx):
        n = 256
        ta = lib.ggml_new_tensor_1d(ctx, lib.GGML_TYPE_F32, n)
        tb = lib.ggml_new_tensor_1d(ctx, lib.GGML_TYPE_F32, n)
        tsum = lib.ggml_add(ctx, ta, tb)
        assert tsum.type == lib.GGML_TYPE_F32

        gf = ffi.new('struct ggml_cgraph*')
        lib.ggml_build_forward_expand(gf, tsum)

        a = np.arange(0, n, dtype=np.float32)
        b = np.arange(n, 0, -1, dtype=np.float32)
        copy(a, ta)
        copy(b, tb)

        lib.ggml_graph_compute_with_ctx(ctx, gf, 1)

        assert np.allclose(numpy(tsum, allow_copy=True), a + b)

class TestQuantization:

    def test_quantized_add(self, ctx):
        n = 256
        ta = lib.ggml_new_tensor_1d(ctx, lib.GGML_TYPE_Q5_K, n)
        tb = lib.ggml_new_tensor_1d(ctx, lib.GGML_TYPE_F32, n)
        tsum = lib.ggml_add(ctx, ta, tb)
        assert tsum.type == lib.GGML_TYPE_Q5_K

        gf = ffi.new('struct ggml_cgraph*')
        lib.ggml_build_forward_expand(gf, tsum)

        a = np.arange(0, n, dtype=np.float32)
        b = np.arange(n, 0, -1, dtype=np.float32)
        copy(a, ta)
        copy(b, tb)

        lib.ggml_graph_compute_with_ctx(ctx, gf, 1)

        unquantized_sum = a + b
        sum = numpy(tsum, allow_copy=True)

        diff = np.linalg.norm(unquantized_sum - sum, np.inf)
        assert diff > 4
        assert diff < 5
