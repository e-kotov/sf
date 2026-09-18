use std::sync::OnceLock;
use wgpu::util::DeviceExt;

struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    euclidean_pipeline: wgpu::ComputePipeline,
    haversine_pipeline: wgpu::ComputePipeline,
    s2_chord_pipeline: wgpu::ComputePipeline,
    vincenty_pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

static GPU_CTX: OnceLock<Option<GpuContext>> = OnceLock::new();

// 1. Planar Euclidean Distance (Flat projection / Cartesian)
const EUCLIDEAN_SHADER_SRC: &str = r#"
struct Params {
    n1: u32,
    n2: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> pts1: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> pts2: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> out_dist: array<f32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let j = global_id.y;
    if (i >= params.n1 || j >= params.n2) { return; }
    let p1 = pts1[i];
    let p2 = pts2[j];
    out_dist[i + j * params.n1] = distance(p1, p2);
}
"#;

// 2. Haversine Great-Circle Geodetic Distance (RAPIDS cuSpatial & GeoRust / Apache Sedona)
// Mean authalic spherical radius: 6,371,008.8m
const HAVERSINE_SHADER_SRC: &str = r#"
struct Params {
    n1: u32,
    n2: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> pts1: array<vec2<f32>>; // [lon, lat] deg
@group(0) @binding(2) var<storage, read> pts2: array<vec2<f32>>; // [lon, lat] deg
@group(0) @binding(3) var<storage, read_write> out_dist: array<f32>; // meters

const EARTH_RADIUS: f32 = 6371008.8;
const DEG_TO_RAD: f32 = 0.017453292519943295;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let j = global_id.y;
    if (i >= params.n1 || j >= params.n2) { return; }
    let p1 = pts1[i];
    let p2 = pts2[j];

    let lat1 = p1.y * DEG_TO_RAD;
    let lat2 = p2.y * DEG_TO_RAD;
    let dlat = (p2.y - p1.y) * DEG_TO_RAD;
    let dlon = (p2.x - p1.x) * DEG_TO_RAD;

    let sin_dlat_2 = sin(dlat * 0.5);
    let sin_dlon_2 = sin(dlon * 0.5);

    let a = sin_dlat_2 * sin_dlat_2 + cos(lat1) * cos(lat2) * sin_dlon_2 * sin_dlon_2;
    let c = 2.0 * asin(clamp(sqrt(a), 0.0, 1.0));
    out_dist[i + j * params.n1] = c * EARTH_RADIUS;
}
"#;

// 3. Google S2 Spherical 3D Unit Vector Chord Angle (R sf default via s2::s2_distance_matrix)
const S2_CHORD_SHADER_SRC: &str = r#"
struct Params {
    n1: u32,
    n2: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> pts1: array<vec2<f32>>; // [lon, lat] deg
@group(0) @binding(2) var<storage, read> pts2: array<vec2<f32>>; // [lon, lat] deg
@group(0) @binding(3) var<storage, read_write> out_dist: array<f32>; // meters

const EARTH_RADIUS: f32 = 6371008.8;
const DEG_TO_RAD: f32 = 0.017453292519943295;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let j = global_id.y;
    if (i >= params.n1 || j >= params.n2) { return; }
    let p1 = pts1[i];
    let p2 = pts2[j];

    let phi1 = p1.y * DEG_TO_RAD;
    let lam1 = p1.x * DEG_TO_RAD;
    let v1 = vec3<f32>(cos(phi1) * cos(lam1), cos(phi1) * sin(lam1), sin(phi1));

    let phi2 = p2.y * DEG_TO_RAD;
    let lam2 = p2.x * DEG_TO_RAD;
    let v2 = vec3<f32>(cos(phi2) * cos(lam2), cos(phi2) * sin(lam2), sin(phi2));

    let chord = distance(v1, v2);
    let central_angle = 2.0 * asin(clamp(chord * 0.5, 0.0, 1.0));
    out_dist[i + j * params.n1] = central_angle * EARTH_RADIUS;
}
"#;

// 4. Vincenty Inverse Geodesic on WGS84 Ellipsoid (Oblate Spheroid)
// Semi-major a = 6378137.0m, Semi-minor b = 6356752.314245m, flattening f = 1/298.257223563
const VINCENTY_SHADER_SRC: &str = r#"
struct Params {
    n1: u32,
    n2: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> pts1: array<vec2<f32>>; // [lon, lat] deg
@group(0) @binding(2) var<storage, read> pts2: array<vec2<f32>>; // [lon, lat] deg
@group(0) @binding(3) var<storage, read_write> out_dist: array<f32>; // meters

const A: f32 = 6378137.0;
const B: f32 = 6356752.314245;
const F: f32 = 0.0033528106647474805; // 1 / 298.257223563
const DEG_TO_RAD: f32 = 0.017453292519943295;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let j = global_id.y;
    if (i >= params.n1 || j >= params.n2) { return; }
    let p1 = pts1[i];
    let p2 = pts2[j];

    // Same point check
    if (abs(p1.x - p2.x) < 1e-6 && abs(p1.y - p2.y) < 1e-6) {
        out_dist[i + j * params.n1] = 0.0;
        return;
    }

    let phi1 = p1.y * DEG_TO_RAD;
    let phi2 = p2.y * DEG_TO_RAD;
    let L = (p2.x - p1.x) * DEG_TO_RAD;

    let tan_u1 = (1.0 - F) * tan(phi1);
    let tan_u2 = (1.0 - F) * tan(phi2);
    let cos_u1 = 1.0 / sqrt(1.0 + tan_u1 * tan_u1);
    let sin_u1 = tan_u1 * cos_u1;
    let cos_u2 = 1.0 / sqrt(1.0 + tan_u2 * tan_u2);
    let sin_u2 = tan_u2 * cos_u2;

    var lambda = L;
    var sin_sigma = 0.0;
    var cos_sigma = 0.0;
    var sigma = 0.0;
    var sin_alpha = 0.0;
    var cos2_alpha = 0.0;
    var cos_2sigma_m = 0.0;

    // Unrolled fixed 6 iterations on GPU (converges in 4-5 iterations)
    for (var iter = 0; iter < 6; iter++) {
        let sin_lambda = sin(lambda);
        let cos_lambda = cos(lambda);

        let term1 = cos_u2 * sin_lambda;
        let term2 = cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_lambda;
        sin_sigma = sqrt(term1 * term1 + term2 * term2);

        if (sin_sigma < 1e-6) {
            out_dist[i + j * params.n1] = 0.0;
            return;
        }

        cos_sigma = sin_u1 * sin_u2 + cos_u1 * cos_u2 * cos_lambda;
        sigma = atan2(sin_sigma, cos_sigma);
        sin_alpha = (cos_u1 * cos_u2 * sin_lambda) / sin_sigma;
        cos2_alpha = 1.0 - sin_alpha * sin_alpha;

        if (cos2_alpha != 0.0) {
            cos_2sigma_m = cos_sigma - (2.0 * sin_u1 * sin_u2) / cos2_alpha;
        } else {
            cos_2sigma_m = 0.0;
        }

        let C = (F / 16.0) * cos2_alpha * (4.0 + F * (4.0 - 3.0 * cos2_alpha));
        lambda = L + (1.0 - C) * F * sin_alpha * (sigma + C * sin_sigma * (cos_2sigma_m + C * cos_sigma * (-1.0 + 2.0 * cos_2sigma_m * cos_2sigma_m)));
    }

    let u_sq = cos2_alpha * ((A * A - B * B) / (B * B));
    let A_factor = 1.0 + (u_sq / 16384.0) * (4096.0 + u_sq * (-768.0 + u_sq * (320.0 - 175.0 * u_sq)));
    let B_factor = (u_sq / 1024.0) * (256.0 + u_sq * (-128.0 + u_sq * (74.0 - 47.0 * u_sq)));

    let delta_sigma = B_factor * sin_sigma * (cos_2sigma_m + (B_factor / 4.0) * (cos_sigma * (-1.0 + 2.0 * cos_2sigma_m * cos_2sigma_m) - (B_factor / 6.0) * cos_2sigma_m * (-3.0 + 4.0 * sin_sigma * sin_sigma) * (-3.0 + 4.0 * cos_2sigma_m * cos_2sigma_m)));

    let s = B * A_factor * (sigma - delta_sigma);
    out_dist[i + j * params.n1] = s;
}
"#;

fn init_gpu() -> Option<GpuContext> {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            })
            .await
            .ok()?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("sf_wgpu_device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                default_queue: wgpu::QueueDescriptor { label: None },
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .ok()?;

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("distance_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("distance_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let create_pipeline = |name: &str, src: &str| {
            let sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(name),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: Some(&pipeline_layout),
                module: &sm,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let euclidean_pipeline = create_pipeline("euclidean", EUCLIDEAN_SHADER_SRC);
        let haversine_pipeline = create_pipeline("haversine", HAVERSINE_SHADER_SRC);
        let s2_chord_pipeline = create_pipeline("s2_chord", S2_CHORD_SHADER_SRC);
        let vincenty_pipeline = create_pipeline("vincenty", VINCENTY_SHADER_SRC);

        Some(GpuContext {
            device,
            queue,
            euclidean_pipeline,
            haversine_pipeline,
            s2_chord_pipeline,
            vincenty_pipeline,
            bind_group_layout,
        })
    })
}

fn get_gpu_context() -> Option<&'static GpuContext> {
    GPU_CTX.get_or_init(init_gpu).as_ref()
}

pub fn run_compute_distance(
    method: i32, // 0 = euclidean, 1 = haversine, 2 = s2_chord, 3 = vincenty_spheroid
    n1: usize,
    pts1: &[f32],
    n2: usize,
    pts2: &[f32],
    out: &mut [f32],
) -> Result<(), &'static str> {
    if pts1.len() < 2 * n1 || pts2.len() < 2 * n2 || out.len() < n1 * n2 {
        return Err("Buffer length mismatch");
    }
    if n1 == 0 || n2 == 0 {
        return Ok(());
    }

    let ctx = get_gpu_context().ok_or("Failed to initialize wgpu device")?;
    let device = &ctx.device;
    let queue = &ctx.queue;

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct Params {
        n1: u32,
        n2: u32,
    }

    let params = Params {
        n1: n1 as u32,
        n2: n2 as u32,
    };

    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params_buf"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let pts1_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("pts1_buf"),
        contents: bytemuck::cast_slice(&pts1[..2 * n1]),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let pts2_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("pts2_buf"),
        contents: bytemuck::cast_slice(&pts2[..2 * n2]),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let out_byte_size = (n1 * n2 * std::mem::size_of::<f32>()) as u64;

    let out_storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("out_storage_buf"),
        size: out_byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging_buf"),
        size: out_byte_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("distance_bind_group"),
        layout: &ctx.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: pts1_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: pts2_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: out_storage_buffer.as_entire_binding() },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("distance_compute_pass"),
            timestamp_writes: None,
        });

        let pipeline = match method {
            1 => &ctx.haversine_pipeline,
            2 => &ctx.s2_chord_pipeline,
            3 => &ctx.vincenty_pipeline,
            _ => &ctx.euclidean_pipeline,
        };

        cpass.set_pipeline(pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        let workgroups_x = (n1 as u32 + 15) / 16;
        let workgroups_y = (n2 as u32 + 15) / 16;
        cpass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }

    encoder.copy_buffer_to_buffer(&out_storage_buffer, 0, &staging_buffer, 0, out_byte_size);
    queue.submit(Some(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    buffer_slice.map_async(wgpu::MapMode::Read, |_| {});

    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|_| "GPU poll failed")?;

    {
        let data = buffer_slice.get_mapped_range().map_err(|_| "Failed to map range")?;
        let result_slice: &[f32] = bytemuck::cast_slice(&data);
        out[..n1 * n2].copy_from_slice(&result_slice[..n1 * n2]);
    }

    staging_buffer.unmap();
    Ok(())
}

#[no_mangle]
pub extern "C" fn c_wgpu_compute(
    method_ptr: *const i32,
    n1_ptr: *const i32,
    pts1_ptr: *const f32,
    n2_ptr: *const i32,
    pts2_ptr: *const f32,
    out_ptr: *mut f32,
) -> i32 {
    if method_ptr.is_null() || n1_ptr.is_null() || pts1_ptr.is_null() || n2_ptr.is_null() || pts2_ptr.is_null() || out_ptr.is_null() {
        return -1;
    }
    let method = unsafe { *method_ptr };
    let n1 = unsafe { *n1_ptr } as usize;
    let n2 = unsafe { *n2_ptr } as usize;
    let pts1 = unsafe { std::slice::from_raw_parts(pts1_ptr, 2 * n1) };
    let pts2 = unsafe { std::slice::from_raw_parts(pts2_ptr, 2 * n2) };
    let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, n1 * n2) };

    match run_compute_distance(method, n1, pts1, n2, pts2, out) {
        Ok(()) => 0,
        Err(_) => -2,
    }
}

// Backward-compatible entry points:
#[no_mangle]
pub extern "C" fn c_wgpu_distance_matrix(
    n1_ptr: *const i32, pts1_ptr: *const f32, n2_ptr: *const i32, pts2_ptr: *const f32, out_ptr: *mut f32,
) -> i32 {
    let method = 0;
    c_wgpu_compute(&method, n1_ptr, pts1_ptr, n2_ptr, pts2_ptr, out_ptr)
}

#[no_mangle]
pub extern "C" fn c_wgpu_haversine_matrix(
    n1_ptr: *const i32, pts1_ptr: *const f32, n2_ptr: *const i32, pts2_ptr: *const f32, out_ptr: *mut f32,
) -> i32 {
    let method = 1;
    c_wgpu_compute(&method, n1_ptr, pts1_ptr, n2_ptr, pts2_ptr, out_ptr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_algorithms() {
        // London (-0.1276, 51.5074) to Paris (2.3522, 48.8566)
        let pts1 = vec![-0.1276f32, 51.5074];
        let pts2 = vec![2.3522f32, 48.8566];
        let mut out = vec![0.0f32; 1];

        // 1. Haversine (cuSpatial / GeoRust / Sedona)
        run_compute_distance(1, 1, &pts1, 1, &pts2, &mut out).unwrap();
        let haversine_km = out[0] / 1000.0;

        // 2. S2 Chord (Google S2 / sf default)
        run_compute_distance(2, 1, &pts1, 1, &pts2, &mut out).unwrap();
        let s2_km = out[0] / 1000.0;

        // 3. Vincenty Ellipsoid (WGS84 Spheroid / Sedona Spheroid)
        run_compute_distance(3, 1, &pts1, 1, &pts2, &mut out).unwrap();
        let vincenty_km = out[0] / 1000.0;

        println!("London to Paris comparison:");
        println!("  Haversine (cuSpatial/Sedona):  {:.3} km", haversine_km);
        println!("  S2 Chord  (Google S2 / sf):    {:.3} km", s2_km);
        println!("  Vincenty  (WGS84 Spheroid):    {:.3} km", vincenty_km);

        assert!((haversine_km - 343.5).abs() < 2.0);
        assert!((s2_km - 343.5).abs() < 2.0);
        assert!((vincenty_km - 343.5).abs() < 2.0);
    }
}
