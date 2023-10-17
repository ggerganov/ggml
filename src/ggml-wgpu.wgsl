struct TensorDimensionParams {
        ne00 : i64;
        ne01 : i64;
        ne02 : i64;
        ne03 : i64;

        nb00 : u64;
        nb01 : u64;
        nb02 : u64;
        nb03 : u64;

        ne10 : i64;
        ne11 : i64;
        ne12 : i64;
        ne13 : i64;

        nb10 : u64;
        nb11 : u64;
        nb12 : u64;
        nb13 : u64;

        ne0 : i64;
        ne1 : i64;
        ne2 : i64;
        ne3 : i64;

        nb0 : u64;
        nb1 : u64;
        nb2 : u64;
        nb3 : u64;
}



@group(0) @binding(0)
var<storage,read_write> src0: array<f32>;

@group(0) @binding(1)
var<storage,read_write> src1: array<f32>;

@group(0) @binding(2)
var<storage,read_write> dst: array<f32>;

@group(0) @binding(3)
var<uniform> tensor_dimension_params: TensorDimensionParams;

