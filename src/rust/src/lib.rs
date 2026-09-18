use std::sync::OnceLock;
use wgpu::util::DeviceExt;

struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    euclidean_pipeline: wgpu::ComputePipeline,
    haversine_pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

static GPU_CTX: OnceLock<Option<GpuContext>> = OnceLock::new();

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
    if (i >= params.n1 || j >= params.n2) {
        return;
    }
    let p1 = pts1[i];
    let p2 = pts2[j];
    let d = distance(p1, p2);
    out_dist[i + j * params.n1] = d;
}
"#;

// Haversine Great-Circle Geodetic Distance on WGS84 Sphere (radius = 6371008.8m)
// Derived from standard formulas used in Apache Sedona and RAPIDS cuSpatial (Apache-2.0)
const HAVERSINE_SHADER_SRC: &str = r#"
struct Params {
    n1: u32,
    n2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> pts1: array<vec2<f32>>; // [lon, lat] in degrees
@group(0) @binding(2) var<storage, read> pts2: array<vec2<f32>>; // [lon, lat] in degrees
@group(0) @binding(3) var<storage, read_write> out_dist: array<f32>; // meters

const EARTH_RADIUS: f32 = 6371008.8; // WGS84 Authalic mean radius in meters
const DEG_TO_RAD: f32 = 0.017453292519943295;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let j = global_id.y;
    if (i >= params.n1 || j >= params.n2) {
        return;
    }
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

        let euclidean_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("euclidean_shader"),
            source: wgpu::ShaderSource::Wgsl(EUCLIDEAN_SHADER_SRC.into()),
        });

        let euclidean_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("euclidean_pipeline"),
            layout: Some(&pipeline_layout),
            module: &euclidean_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let haversine_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("haversine_shader"),
            source: wgpu::ShaderSource::Wgsl(HAVERSINE_SHADER_SRC.into()),
        });

        let haversine_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("haversine_pipeline"),
            layout: Some(&pipeline_layout),
            module: &haversine_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Some(GpuContext {
            device,
            queue,
            euclidean_pipeline,
            haversine_pipeline,
            bind_group_layout,
        })
    })
}

fn get_gpu_context() -> Option<&'static GpuContext> {
    GPU_CTX.get_or_init(init_gpu).as_ref()
}

pub fn run_compute_distance(
    use_haversine: bool,
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
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: pts1_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: pts2_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: out_storage_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("distance_encoder"),
    });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("distance_compute_pass"),
            timestamp_writes: None,
        });
        let pipeline = if use_haversine {
            &ctx.haversine_pipeline
        } else {
            &ctx.euclidean_pipeline
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
        let data = buffer_slice
            .get_mapped_range()
            .map_err(|_| "Failed to map range")?;
        let result_slice: &[f32] = bytemuck::cast_slice(&data);
        out[..n1 * n2].copy_from_slice(&result_slice[..n1 * n2]);
    }

    staging_buffer.unmap();
    Ok(())
}

#[no_mangle]
pub extern "C" fn c_wgpu_distance_matrix(
    n1_ptr: *const i32,
    pts1_ptr: *const f32,
    n2_ptr: *const i32,
    pts2_ptr: *const f32,
    out_ptr: *mut f32,
) -> i32 {
    if n1_ptr.is_null() || pts1_ptr.is_null() || n2_ptr.is_null() || pts2_ptr.is_null() || out_ptr.is_null() {
        return -1;
    }
    let n1 = unsafe { *n1_ptr } as usize;
    let n2 = unsafe { *n2_ptr } as usize;
    let pts1 = unsafe { std::slice::from_raw_parts(pts1_ptr, 2 * n1) };
    let pts2 = unsafe { std::slice::from_raw_parts(pts2_ptr, 2 * n2) };
    let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, n1 * n2) };

    match run_compute_distance(false, n1, pts1, n2, pts2, out) {
        Ok(()) => 0,
        Err(_) => -2,
    }
}

#[no_mangle]
pub extern "C" fn c_wgpu_haversine_matrix(
    n1_ptr: *const i32,
    pts1_ptr: *const f32,
    n2_ptr: *const i32,
    pts2_ptr: *const f32,
    out_ptr: *mut f32,
) -> i32 {
    if n1_ptr.is_null() || pts1_ptr.is_null() || n2_ptr.is_null() || pts2_ptr.is_null() || out_ptr.is_null() {
        return -1;
    }
    let n1 = unsafe { *n1_ptr } as usize;
    let n2 = unsafe { *n2_ptr } as usize;
    let pts1 = unsafe { std::slice::from_raw_parts(pts1_ptr, 2 * n1) };
    let pts2 = unsafe { std::slice::from_raw_parts(pts2_ptr, 2 * n2) };
    let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, n1 * n2) };

    match run_compute_distance(true, n1, pts1, n2, pts2, out) {
        Ok(()) => 0,
        Err(_) => -2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_haversine() {
        // Point 1: London (lon -0.1276, lat 51.5074)
        // Point 2: Paris (lon 2.3522, lat 48.8566)
        let pts1 = vec![-0.1276f32, 51.5074];
        let pts2 = vec![2.3522f32, 48.8566];
        let mut out = vec![0.0f32; 1];
        let res = run_compute_distance(true, 1, &pts1, 1, &pts2, &mut out);
        assert!(res.is_ok());
        // London to Paris distance is ~343.5 km (343,500 meters)
        let dist_km = out[0] / 1000.0;
        println!("London to Paris distance: {:.2} km", dist_km);
        assert!((dist_km - 343.5).abs() < 2.0);
    }
}
