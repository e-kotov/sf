use std::sync::OnceLock;
use wgpu::util::DeviceExt;

struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

static GPU_CTX: OnceLock<Option<GpuContext>> = OnceLock::new();

const SHADER_SRC: &str = r#"
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
    // R column-major order: index = i + j * n1
    out_dist[i + j * params.n1] = d;
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
                required_limits: wgpu::Limits::downlevel_defaults(),
                default_queue: wgpu::QueueDescriptor { label: None },
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .ok()?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("distance_matrix_shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("distance_bind_group_layout"),
            entries: &[
                // params
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
                // pts1
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
                // pts2
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
                // out_dist
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

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("distance_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Some(GpuContext {
            device,
            queue,
            pipeline,
            bind_group_layout,
        })
    })
}

fn get_gpu_context() -> Option<&'static GpuContext> {
    GPU_CTX.get_or_init(init_gpu).as_ref()
}

pub fn compute_distance_matrix(
    n1: usize,
    pts1: &[f32], // length 2 * n1
    n2: usize,
    pts2: &[f32], // length 2 * n2
    out: &mut [f32], // length n1 * n2
) -> Result<(), &'static str> {
    if pts1.len() < 2 * n1 || pts2.len() < 2 * n2 || out.len() < n1 * n2 {
        return Err("Buffer length mismatch");
    }

    if n1 == 0 || n2 == 0 {
        return Ok(());
    }

    let ctx = get_gpu_context().ok_or("Failed to initialize wgpu / Metal device")?;
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
        cpass.set_pipeline(&ctx.pipeline);
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
pub extern "C" fn c_wgpu_is_available() -> i32 {
    if get_gpu_context().is_some() {
        1
    } else {
        0
    }
}

/// C ABI entry point for R:
/// pts1 is flat array of interleaved [x0, y0, x1, y1, ...] length 2 * n1
/// pts2 is flat array of interleaved [x0, y0, x1, y1, ...] length 2 * n2
/// out is allocated matrix of length n1 * n2 in column-major order
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

    match compute_distance_matrix(n1, pts1, n2, pts2, out) {
        Ok(()) => 0,
        Err(_) => -2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_matrix() {
        // Point 1: (0, 0), Point 2: (3, 4)
        let pts1 = vec![0.0f32, 0.0, 3.0, 4.0];
        // Point 1: (0, 0), Point 2: (0, 4)
        let pts2 = vec![0.0f32, 0.0, 0.0, 4.0];

        let mut out = vec![0.0f32; 4];
        let res = compute_distance_matrix(2, &pts1, 2, &pts2, &mut out);
        assert!(res.is_ok(), "Distance matrix calculation failed: {:?}", res);

        // dist((0,0), (0,0)) = 0
        // dist((3,4), (0,0)) = 5
        // dist((0,0), (0,4)) = 4
        // dist((3,4), (0,4)) = 3
        // Column-major: out[0] = d(p1_0, p2_0) = 0
        //              out[1] = d(p1_1, p2_0) = 5
        //              out[2] = d(p1_0, p2_1) = 4
        //              out[3] = d(p1_1, p2_1) = 3
        assert!((out[0] - 0.0).abs() < 1e-5);
        assert!((out[1] - 5.0).abs() < 1e-5);
        assert!((out[2] - 4.0).abs() < 1e-5);
        assert!((out[3] - 3.0).abs() < 1e-5);
    }
}
