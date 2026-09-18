#[test]
fn test_metal_limits() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).unwrap();
    let info = adapter.get_info();
    let limits = adapter.limits();
    println!("Backend: {:?}", info.backend);
    println!("Adapter name: {}", info.name);
    println!("Max buffer size: {} MB", limits.max_buffer_size / (1024 * 1024));
    println!("Max storage buffer binding size: {} MB", limits.max_storage_buffer_binding_size / (1024 * 1024));
}
