use anyhow::Result;
use shared_structures::SharedRingBuffer;
use std::env;
use std::ffi::c_void;
use std::mem::MaybeUninit;
use std::num::NonZero;
use std::os::fd::AsRawFd as _;
use std::ptr::NonNull;
use std::sync::Arc;
use std::time::Duration;

use libc;
use xcb::{self, Xid, x};

use raw_window_handle::{
    DisplayHandle, HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle,
    WindowHandle, XcbDisplayHandle, XcbWindowHandle,
};

use xbar_core::{
    AppState, BarConfig, Color, SHARED_TOKEN, ShapeStyle, ThemeMode, arm_second_timer,
    cairo::{self, Context, Format, ImageSurface},
    colors_for_theme, draw_bar, initialize_logging,
    pango::FontDescription,
    spawn_shared_eventfd_notifier,
};

// ============================================================================
// 1. RAW WINDOW HANDLE 用于 WGPU 识别 XCB
// ============================================================================
struct XcbTarget {
    conn: *mut c_void,
    window: u32,
}

// 解决 *mut c_void 的跨线程传递问题
unsafe impl Send for XcbTarget {}
unsafe impl Sync for XcbTarget {}

impl HasDisplayHandle for XcbTarget {
    fn display_handle(&self) -> Result<DisplayHandle<'_>, raw_window_handle::HandleError> {
        let handle = XcbDisplayHandle::new(Some(NonNull::new(self.conn).unwrap()), 0);
        Ok(unsafe { DisplayHandle::borrow_raw(RawDisplayHandle::Xcb(handle)) })
    }
}

impl HasWindowHandle for XcbTarget {
    fn window_handle(&self) -> Result<WindowHandle<'_>, raw_window_handle::HandleError> {
        let handle = XcbWindowHandle::new(std::num::NonZeroU32::new(self.window).unwrap());
        Ok(unsafe { WindowHandle::borrow_raw(RawWindowHandle::Xcb(handle)) })
    }
}

// ============================================================================
// 2. WGPU 封装
// ============================================================================
struct Gpu {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    cpu_tex: wgpu::Texture,
    cpu_tex_view: wgpu::TextureView,
    cpu_tex_format: wgpu::TextureFormat,
    sampler: wgpu::Sampler,
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    width: u32,
    height: u32,
}

const FULLSCREEN_WGSL: &str = r#"
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vid: u32) -> VSOut {
  var pos = array<vec2<f32>, 3>(
    vec2(-1.0, -1.0),
    vec2( 3.0, -1.0),
    vec2(-1.0,  3.0),
  )[vid];
  var out: VSOut;
  out.pos = vec4(pos, 0.0, 1.0);
  let uv = 0.5 * pos + vec2(0.5, 0.5);
  out.uv = vec2(uv.x, 1.0 - uv.y);
  return out;
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4<f32> {
  return textureSample(tex, samp, in.uv);
}
"#;

impl Gpu {
    async fn new(target: Arc<XcbTarget>, width: u32, height: u32) -> Result<Self> {
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(target)?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("No adapter found");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await?;

        let caps = surface.get_capabilities(&adapter);
        let surface_format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let cpu_tex_format = if surface_format == wgpu::TextureFormat::Bgra8UnormSrgb {
            wgpu::TextureFormat::Bgra8UnormSrgb
        } else {
            wgpu::TextureFormat::Rgba8UnormSrgb
        };

        let cpu_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("cpu-upload-texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: cpu_tex_format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let cpu_tex_view = cpu_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fullscreen-shader"),
            source: wgpu::ShaderSource::Wgsl(FULLSCREEN_WGSL.into()),
        });

        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tex-sampler-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline-layout"),
            bind_group_layouts: &[Some(&bind_layout)], // 修复：包装在 Some 中
            immediate_size: 0,                         // 修复：添加 immediate_size
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("fullscreen-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            cache: None,
            multiview_mask: NonZero::new(0),
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tex-sampler-bindgroup"),
            layout: &bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&cpu_tex_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        Ok(Self {
            surface,
            device,
            queue,
            config,
            cpu_tex,
            cpu_tex_view,
            cpu_tex_format,
            sampler,
            pipeline,
            bind_group,
            width,
            height,
        })
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.width = width;
        self.height = height;
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);

        self.cpu_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("cpu-upload-texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.cpu_tex_format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.cpu_tex_view = self
            .cpu_tex
            .create_view(&wgpu::TextureViewDescriptor::default());

        let bind_layout = self.pipeline.get_bind_group_layout(0);
        self.bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tex-sampler-bindgroup"),
            layout: &bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.cpu_tex_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
    }

    fn upload_and_present(&self, cpu_data: &[u8], stride: u32) -> Result<()> {
        let bpr = stride;
        let aligned_bpr = ((bpr + 255) / 256) * 256;
        let height = self.height;

        let mut padded: Vec<u8>;
        let data_ref: &[u8] = if aligned_bpr == bpr {
            cpu_data
        } else {
            padded = vec![0u8; aligned_bpr as usize * height as usize];
            for y in 0..height as usize {
                let src = &cpu_data[y * bpr as usize..(y + 1) * bpr as usize];
                let dst =
                    &mut padded[y * aligned_bpr as usize..y * aligned_bpr as usize + bpr as usize];
                dst.copy_from_slice(src);
            }
            &padded
        };

        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.cpu_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data_ref,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(aligned_bpr),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );

        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame) => frame,
            wgpu::CurrentSurfaceTexture::Suboptimal(frame) => {
                self.surface.configure(&self.device, &self.config);
                frame
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                self.surface.configure(&self.device, &self.config);
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Lost => anyhow::bail!("wgpu surface lost"),
            wgpu::CurrentSurfaceTexture::Validation => {
                anyhow::bail!("wgpu surface validation error")
            }
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });
            rp.set_pipeline(&self.pipeline);
            rp.set_bind_group(0, &self.bind_group, &[]);
            rp.draw(0..3, 0..1);
        }

        self.queue.submit(Some(encoder.finish()));
        frame.present();
        Ok(())
    }
}

// ============================================================================
// 3. 辅助功能 (主题、重绘、事件处理)
// ============================================================================
fn tuned_colors_for_theme(mode: ThemeMode) -> xbar_core::Colors {
    let mut c = colors_for_theme(mode);
    match mode {
        ThemeMode::Dark => {
            c.bg = Color::rgb(13, 16, 23);
            c.text = Color::rgb(235, 238, 245);
        }
        ThemeMode::Light => {
            c.bg = Color::rgb(246, 247, 250);
            c.text = Color::rgb(22, 24, 28);
        }
    }
    c
}

fn intern_atom(conn: &xcb::Connection, name: &'static [u8]) -> Result<x::Atom> {
    let cookie = conn.send_request(&x::InternAtom {
        only_if_exists: false,
        name,
    });
    Ok(conn.wait_for_reply(cookie)?.atom())
}

fn set_net_wm_name(conn: &xcb::Connection, window: x::Window, name: &str) -> Result<()> {
    let utf8_string = intern_atom(conn, b"UTF8_STRING")?;
    let net_wm_name = intern_atom(conn, b"_NET_WM_NAME")?;

    conn.send_and_check_request(&x::ChangeProperty {
        mode: x::PropMode::Replace,
        window,
        property: net_wm_name,
        r#type: utf8_string,
        data: name.as_bytes(),
    })?;

    Ok(())
}

fn refresh_runtime_state(state: &mut AppState) -> bool {
    let mut changed = false;
    let new_time = state.format_time();
    if new_time != state.last_time_string {
        state.last_time_string = new_time;
        changed = true;
    }
    if state.last_monitor_update.elapsed() >= Duration::from_secs(2) {
        changed |= state.system_monitor.update_if_needed();
        changed |= state.audio_manager.update_if_needed();
        state.last_monitor_update = std::time::Instant::now();
    }
    changed
}

fn sync_shared_state(state: &mut AppState) -> bool {
    let Some(shared_buffer) = state.shared_buffer.as_ref().cloned() else {
        return false;
    };

    match shared_buffer.try_read_latest_message() {
        Ok(Some(msg)) => {
            state.update_from_shared(msg);
            true
        }
        Ok(None) | Err(_) => false,
    }
}

fn redraw(
    gpu: &Gpu,
    cpu_frame: &mut Vec<u8>,
    width: u16,
    height: u16,
    colors: &xbar_core::Colors,
    state: &mut AppState,
    font: &FontDescription,
    cfg: &BarConfig,
) -> Result<()> {
    let w = width as i32;
    let h = height as i32;
    let stride = cairo::Format::ARgb32.stride_for_width(w as u32).unwrap();

    let needed = (stride * h) as usize;
    if cpu_frame.len() != needed {
        cpu_frame.resize(needed, 0);
    }

    let surface = unsafe {
        ImageSurface::create_for_data_unsafe(cpu_frame.as_mut_ptr(), Format::ARgb32, w, h, stride)?
    };
    let cr = Context::new(&surface)?;
    cr.set_source_rgba(0.0, 0.0, 0.0, 1.0);
    cr.set_operator(cairo::Operator::Source);
    cr.paint()?;
    draw_bar(&cr, width, height, colors, state, font, cfg)?;
    surface.flush();

    gpu.upload_and_present(cpu_frame, stride as u32)?;
    Ok(())
}

// ============================================================================
// 4. MAIN LOOP
// ============================================================================
fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let shared_path = args.get(1).cloned().unwrap_or_default();
    initialize_logging("xcb_wgpu_bar", &shared_path)?;

    let shared_buffer = SharedRingBuffer::create_shared_ring_buffer_aux(&shared_path).map(Arc::new);
    let shared_efd = spawn_shared_eventfd_notifier(shared_buffer.clone(), true);

    let (conn, screen_num) = xcb::Connection::connect(None)?;
    let setup = conn.get_setup();
    let screen = setup.roots().nth(screen_num as usize).unwrap();

    let cfg = BarConfig {
        bar_height: 38,
        padding_x: 10.0,
        padding_y: 6.0,
        tag_spacing: 6.0,
        pill_hpadding: 10.0,
        pill_radius: 12.0,
        shape_style: ShapeStyle::Pill,
        time_icon: "🕐",
        screenshot_label: "📸",
        tag_labels: ["🖥", "🌐", "📁", "💬", "📝", "🎵", "⚙", "📊", "🏠"],
        theme_dark_label: "🌙",
        theme_light_label: "☀️",
        monitor_labels: ["🥇", "🥈", "🥉", "❔"],
        volume_label: "🔊",
        mute_label: "🔇",
        show_audio: true,
        show_theme_toggle: true,
        volume_step: 5,
    };

    let win = conn.generate_id();
    let mut current_width = screen.width_in_pixels();
    let mut current_height = cfg.bar_height;

    conn.send_and_check_request(&x::CreateWindow {
        depth: x::COPY_FROM_PARENT as u8,
        wid: win,
        parent: screen.root(),
        x: 0,
        y: 0,
        width: current_width,
        height: current_height,
        border_width: 0,
        class: x::WindowClass::InputOutput,
        visual: screen.root_visual(),
        value_list: &[
            x::Cw::BackPixmap(x::Pixmap::none()),
            x::Cw::EventMask(
                x::EventMask::EXPOSURE
                    | x::EventMask::STRUCTURE_NOTIFY
                    | x::EventMask::BUTTON_PRESS
                    | x::EventMask::POINTER_MOTION
                    | x::EventMask::ENTER_WINDOW
                    | x::EventMask::LEAVE_WINDOW,
            ),
        ],
    })?;

    set_net_wm_name(&conn, win, env!("CARGO_PKG_NAME"))?;

    // 绑定 WGPU
    let target = Arc::new(XcbTarget {
        conn: conn.get_raw_conn() as *mut c_void,
        window: win.resource_id(),
    });
    let mut gpu = pollster::block_on(Gpu::new(
        target,
        current_width as u32,
        current_height as u32,
    ))?;
    let mut cpu_frame = Vec::new();

    conn.send_and_check_request(&x::MapWindow { window: win })?;
    conn.flush()?;

    let font = FontDescription::from_string(
        &env::var("XBAR_FONT").unwrap_or_else(|_| "monospace 11".into()),
    );
    let mut state = AppState::new(shared_buffer);
    state.theme_mode = ThemeMode::Dark;
    let _ = sync_shared_state(&mut state);
    state.last_time_string = state.format_time();
    let _ = state.system_monitor.update_if_needed();
    let _ = state.audio_manager.update_if_needed();
    state.last_monitor_update = std::time::Instant::now();
    let mut colors = tuned_colors_for_theme(state.theme_mode);

    redraw(
        &gpu,
        &mut cpu_frame,
        current_width,
        current_height,
        &colors,
        &mut state,
        &font,
        &cfg,
    )?;

    let epfd = unsafe { libc::epoll_create1(libc::EPOLL_CLOEXEC) };
    let tfd = unsafe {
        libc::timerfd_create(
            libc::CLOCK_MONOTONIC,
            libc::TFD_NONBLOCK | libc::TFD_CLOEXEC,
        )
    };
    arm_second_timer(tfd)?;

    let mut ev_x = libc::epoll_event {
        events: libc::EPOLLIN as u32,
        u64: 1,
    };
    unsafe { libc::epoll_ctl(epfd, libc::EPOLL_CTL_ADD, conn.as_raw_fd(), &mut ev_x) };
    let mut ev_t = libc::epoll_event {
        events: libc::EPOLLIN as u32,
        u64: 2,
    };
    unsafe { libc::epoll_ctl(epfd, libc::EPOLL_CTL_ADD, tfd, &mut ev_t) };

    if let Some(efd) = shared_efd {
        let mut ev_s = libc::epoll_event {
            events: libc::EPOLLIN as u32,
            u64: SHARED_TOKEN,
        };
        unsafe { libc::epoll_ctl(epfd, libc::EPOLL_CTL_ADD, efd, &mut ev_s) };
    }

    let mut events: [libc::epoll_event; 32] = unsafe { MaybeUninit::zeroed().assume_init() };

    loop {
        let nfds = unsafe { libc::epoll_wait(epfd, events.as_mut_ptr(), 32, -1) };
        if nfds < 0 {
            continue;
        }

        for i in 0..(nfds as usize) {
            match events[i].u64 {
                1 => {
                    while let Ok(Some(event)) = conn.poll_for_event() {
                        let mut need_redraw = false;
                        match event {
                            xcb::Event::X(x::Event::Expose(e)) if e.count() == 0 => {
                                need_redraw = true
                            }
                            xcb::Event::X(x::Event::ConfigureNotify(e)) if e.window() == win => {
                                current_width = e.width() as u16;
                                current_height = e.height() as u16;
                                gpu.resize(current_width as u32, current_height as u32);
                                need_redraw = true;
                            }
                            xcb::Event::X(x::Event::MotionNotify(e)) => {
                                need_redraw = state.update_hover(e.event_x(), e.event_y());
                            }
                            xcb::Event::X(x::Event::ButtonPress(e)) => {
                                let before_theme = state.theme_mode;
                                if state.handle_buttons(e.event_x(), e.event_y(), e.detail().into())
                                {
                                    if state.theme_mode != before_theme {
                                        colors = tuned_colors_for_theme(state.theme_mode);
                                    }
                                    need_redraw = true;
                                }
                            }
                            _ => {}
                        }
                        if need_redraw {
                            let _ = redraw(
                                &gpu,
                                &mut cpu_frame,
                                current_width,
                                current_height,
                                &colors,
                                &mut state,
                                &font,
                                &cfg,
                            );
                        }
                    }
                }
                2 => {
                    let mut buf = [0u8; 8];
                    if unsafe { libc::read(tfd, buf.as_mut_ptr() as _, 8) } == 8
                        && refresh_runtime_state(&mut state)
                    {
                        let _ = redraw(
                            &gpu,
                            &mut cpu_frame,
                            current_width,
                            current_height,
                            &colors,
                            &mut state,
                            &font,
                            &cfg,
                        );
                    }
                }
                SHARED_TOKEN => {
                    if let Some(efd) = shared_efd {
                        let mut buf = [0u8; 8];
                        if unsafe { libc::read(efd, buf.as_mut_ptr() as _, 8) } == 8
                            && sync_shared_state(&mut state)
                        {
                            let _ = redraw(
                                &gpu,
                                &mut cpu_frame,
                                current_width,
                                current_height,
                                &colors,
                                &mut state,
                                &font,
                                &cfg,
                            );
                        }
                    }
                }
                _ => {}
            }
        }
    }
}
