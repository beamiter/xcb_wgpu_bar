use anyhow::{Result, anyhow};
use cairo::{Context, Format, ImageSurface};
use pango::FontDescription;
use std::cell::{Cell, RefCell};
use std::env;
use std::ffi::c_void;
use std::io;
use std::num::NonZero;
use std::os::fd::{AsRawFd as _, FromRawFd as _, OwnedFd, RawFd};
use std::ptr::NonNull;
use std::sync::Arc;
use std::time::Duration;

use xcb::{self, Xid, x};

use raw_window_handle::{
    DisplayHandle, HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle,
    WindowHandle, XcbDisplayHandle, XcbWindowHandle,
};

use xbar_core::linux::AlignedTimer;
use xbar_core::presentation::{Point, PointerAction, PresentationConfig, Size};
use xbar_core::render::cairo::CairoBar;
use xbar_core::{
    BarEffect, BarRuntime, ModelConfig, MonitorGeometry, NotifierChange, PlatformEffectHandler,
    RuntimeUpdate, TransportNotifierSlot, TransportRecoveryConfig,
};
use xbar_linux_actions::ProcessActionHandler;

const X_TOKEN: u64 = 1;
const TIMER_TOKEN: u64 = 2;
const SHARED_TOKEN: u64 = 3;
const TRANSPORT_RETRY_INTERVAL: Duration = Duration::from_secs(2);

// ============================================================================
// 1. RAW WINDOW HANDLE 用于 WGPU 识别 XCB
// ============================================================================
struct XcbTarget {
    conn: *mut c_void,
    screen: i32,
    window: u32,
}

// 解决 *mut c_void 的跨线程传递问题
unsafe impl Send for XcbTarget {}
unsafe impl Sync for XcbTarget {}

impl HasDisplayHandle for XcbTarget {
    fn display_handle(&self) -> Result<DisplayHandle<'_>, raw_window_handle::HandleError> {
        let handle = XcbDisplayHandle::new(Some(NonNull::new(self.conn).unwrap()), self.screen);
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
    upload_scratch: Vec<u8>,
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
                apply_limit_buckets: false,
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
            color_space: wgpu::SurfaceColorSpace::Auto,
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
            upload_scratch: Vec::new(),
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

    fn upload_and_present(&mut self, cpu_data: &[u8], stride: u32) -> Result<()> {
        let bpr = stride;
        let aligned_bpr = bpr.div_ceil(256) * 256;
        let height = self.height;
        let width = self.width;
        let source_row_bytes = bpr as usize;
        let upload_row_bytes = aligned_bpr as usize;
        let height_usize = height as usize;
        let pixel_bytes = (width as usize)
            .checked_mul(4)
            .ok_or_else(|| anyhow!("upload row size overflow"))?;
        if pixel_bytes > source_row_bytes {
            return Err(anyhow!(
                "Cairo stride is smaller than the visible pixel row"
            ));
        }
        let source_len = source_row_bytes
            .checked_mul(height_usize)
            .ok_or_else(|| anyhow!("source frame size overflow"))?;
        if cpu_data.len() < source_len {
            return Err(anyhow!("Cairo frame is shorter than stride * height"));
        }

        let rgba_upload = self.cpu_tex_format == wgpu::TextureFormat::Rgba8UnormSrgb;
        let data_ref: &[u8] = if rgba_upload || aligned_bpr != bpr {
            let upload_len = upload_row_bytes
                .checked_mul(height_usize)
                .ok_or_else(|| anyhow!("upload frame size overflow"))?;
            self.upload_scratch.resize(upload_len, 0);
            for row in 0..height_usize {
                let source_start = row * source_row_bytes;
                let upload_start = row * upload_row_bytes;
                let source = &cpu_data[source_start..source_start + source_row_bytes];
                let upload =
                    &mut self.upload_scratch[upload_start..upload_start + upload_row_bytes];
                if rgba_upload {
                    for (source, upload) in source[..pixel_bytes]
                        .chunks_exact(4)
                        .zip(upload[..pixel_bytes].chunks_exact_mut(4))
                    {
                        upload.copy_from_slice(&[source[2], source[1], source[0], source[3]]);
                    }
                    upload[pixel_bytes..source_row_bytes]
                        .copy_from_slice(&source[pixel_bytes..source_row_bytes]);
                } else {
                    upload[..source_row_bytes].copy_from_slice(source);
                }
                upload[source_row_bytes..].fill(0);
            }
            &self.upload_scratch
        } else {
            cpu_data
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
        self.queue.present(frame);
        Ok(())
    }
}

// ============================================================================
// 3. XCB platform adapter and Cairo-to-wgpu presentation
// ============================================================================
struct Atoms {
    net_wm_window_type: x::Atom,
    net_wm_window_type_dock: x::Atom,
    net_wm_state: x::Atom,
    net_wm_state_above: x::Atom,
    net_wm_desktop: x::Atom,
    net_wm_strut_partial: x::Atom,
    net_wm_strut: x::Atom,
    net_wm_name: x::Atom,
    utf8_string: x::Atom,
    atom: x::Atom,
    cardinal: x::Atom,
}

fn intern_atom(conn: &xcb::Connection, name: &[u8]) -> Result<x::Atom> {
    let cookie = conn.send_request(&x::InternAtom {
        only_if_exists: false,
        name,
    });
    Ok(conn.wait_for_reply(cookie)?.atom())
}

fn intern_atoms(conn: &xcb::Connection) -> Result<Atoms> {
    Ok(Atoms {
        net_wm_window_type: intern_atom(conn, b"_NET_WM_WINDOW_TYPE")?,
        net_wm_window_type_dock: intern_atom(conn, b"_NET_WM_WINDOW_TYPE_DOCK")?,
        net_wm_state: intern_atom(conn, b"_NET_WM_STATE")?,
        net_wm_state_above: intern_atom(conn, b"_NET_WM_STATE_ABOVE")?,
        net_wm_desktop: intern_atom(conn, b"_NET_WM_DESKTOP")?,
        net_wm_strut_partial: intern_atom(conn, b"_NET_WM_STRUT_PARTIAL")?,
        net_wm_strut: intern_atom(conn, b"_NET_WM_STRUT")?,
        net_wm_name: intern_atom(conn, b"_NET_WM_NAME")?,
        utf8_string: intern_atom(conn, b"UTF8_STRING")?,
        atom: intern_atom(conn, b"ATOM")?,
        cardinal: intern_atom(conn, b"CARDINAL")?,
    })
}

fn change_property_32(
    conn: &xcb::Connection,
    win: x::Window,
    property: x::Atom,
    property_type: x::Atom,
    data: &[u32],
) -> Result<()> {
    // The u32 element type makes xcb emit format=32. Converting these values
    // to bytes would silently emit a malformed format=8 EWMH property.
    conn.send_and_check_request(&x::ChangeProperty {
        mode: x::PropMode::Replace,
        window: win,
        property,
        r#type: property_type,
        data,
    })?;
    Ok(())
}

fn change_property_8(
    conn: &xcb::Connection,
    win: x::Window,
    property: x::Atom,
    property_type: x::Atom,
    data: &[u8],
) -> Result<()> {
    conn.send_and_check_request(&x::ChangeProperty {
        mode: x::PropMode::Replace,
        window: win,
        property,
        r#type: property_type,
        data,
    })?;
    Ok(())
}

fn update_strut(
    conn: &xcb::Connection,
    atoms: &Atoms,
    win: x::Window,
    x: i32,
    y: i32,
    width: u32,
    bar_height: u16,
) -> Result<()> {
    let top = u32::try_from(y)
        .unwrap_or(0)
        .saturating_add(u32::from(bar_height));
    let top_start_x = u32::try_from(x).unwrap_or(0);
    let top_end_x = top_start_x.saturating_add(width.saturating_sub(1));
    change_property_32(
        conn,
        win,
        atoms.net_wm_strut_partial,
        atoms.cardinal,
        &[0, 0, top, 0, 0, 0, 0, 0, top_start_x, top_end_x, 0, 0],
    )?;
    change_property_32(
        conn,
        win,
        atoms.net_wm_strut,
        atoms.cardinal,
        &[0, 0, top, 0],
    )
}

fn set_dock_properties(
    conn: &xcb::Connection,
    atoms: &Atoms,
    win: x::Window,
    width: u32,
    bar_height: u16,
) -> Result<()> {
    change_property_32(
        conn,
        win,
        atoms.net_wm_window_type,
        atoms.atom,
        &[atoms.net_wm_window_type_dock.resource_id()],
    )?;
    change_property_32(
        conn,
        win,
        atoms.net_wm_state,
        atoms.atom,
        &[atoms.net_wm_state_above.resource_id()],
    )?;
    change_property_32(conn, win, atoms.net_wm_desktop, atoms.cardinal, &[u32::MAX])?;
    update_strut(conn, atoms, win, 0, 0, width, bar_height)?;
    change_property_8(
        conn,
        win,
        atoms.net_wm_name,
        atoms.utf8_string,
        env!("CARGO_PKG_NAME").as_bytes(),
    )
}

struct WindowAdapter<'a> {
    conn: &'a xcb::Connection,
    screen: &'a x::Screen,
    atoms: &'a Atoms,
    win: x::Window,
    bar_height: Cell<u16>,
    process_actions: RefCell<ProcessActionHandler>,
}

impl WindowAdapter<'_> {
    fn sync_bar_height(&self, bar: &mut CairoBar, height: u16) {
        // A window manager may enforce its configured dock height instead of
        // the size requested when the window was created. Keep both future
        // geometry requests and the presentation viewport fill in sync with
        // that final server-side height.
        self.bar_height.set(height);
        bar.config_mut().bar_height = f32::from(height);
    }

    fn apply_runtime_update(&self, update: RuntimeUpdate) -> Result<bool> {
        let needs_redraw = update.needs_redraw();
        for issue in update.issues {
            log::warn!("xbar runtime issue: {issue:?}");
        }
        for effect in update.platform_effects {
            self.apply_effect(effect)?;
        }
        Ok(needs_redraw)
    }

    fn apply_effect(&self, effect: BarEffect) -> Result<()> {
        match effect {
            BarEffect::ApplyMonitorGeometry(geometry) => self.apply_geometry(geometry),
            BarEffect::ClearMonitorGeometry => self.apply_geometry(MonitorGeometry {
                x: 0,
                y: 0,
                width: u32::from(self.screen.width_in_pixels()),
                height: u32::from(self.screen.height_in_pixels()),
            }),
            effect @ (BarEffect::Screenshot | BarEffect::OpenAudioControl) => {
                self.process_actions.borrow_mut().handle(effect)?;
                Ok(())
            }
            BarEffect::WindowManager(command) => {
                log::warn!("no shared transport handled window-manager command: {command:?}");
                Ok(())
            }
            BarEffect::ToggleMute
            | BarEffect::AdjustVolume(_)
            | BarEffect::AdjustBrightness(_)
            | BarEffect::RefreshBattery => {
                log::warn!("enabled xbar provider returned platform effect: {effect:?}");
                Ok(())
            }
        }
    }

    fn apply_geometry(&self, geometry: MonitorGeometry) -> Result<()> {
        let width = geometry.width.max(1);
        let bar_height = self.bar_height.get();
        self.conn.send_and_check_request(&x::ConfigureWindow {
            window: self.win,
            value_list: &[
                x::ConfigWindow::X(geometry.x),
                x::ConfigWindow::Y(geometry.y),
                x::ConfigWindow::Width(width),
                x::ConfigWindow::Height(u32::from(bar_height)),
            ],
        })?;
        update_strut(
            self.conn, self.atoms, self.win, geometry.x, geometry.y, width, bar_height,
        )?;
        self.conn.flush()?;
        Ok(())
    }
}

fn pointer_action(button: u8) -> Option<PointerAction> {
    match button {
        1 => Some(PointerAction::Primary),
        3 => Some(PointerAction::Secondary),
        4 => Some(PointerAction::ScrollUp),
        5 => Some(PointerAction::ScrollDown),
        _ => None,
    }
}

fn redraw(
    gpu: &mut Gpu,
    cpu_frame: &mut Vec<u8>,
    width: u16,
    height: u16,
    bar: &mut CairoBar,
) -> Result<()> {
    let w = i32::from(width);
    let h = i32::from(height);
    let stride = Format::ARgb32.stride_for_width(u32::from(width))?;

    let needed = (stride * h) as usize;
    if cpu_frame.len() != needed {
        cpu_frame.resize(needed, 0);
    }

    let surface = unsafe {
        ImageSurface::create_for_data_unsafe(cpu_frame.as_mut_ptr(), Format::ARgb32, w, h, stride)?
    };
    let context = Context::new(&surface)?;
    bar.render(&context, Size::new(f32::from(width), f32::from(height)))?;
    let _ = bar.runtime_mut().take_changes();
    surface.flush();

    gpu.upload_and_present(cpu_frame, u32::try_from(stride)?)?;
    Ok(())
}

fn create_epoll() -> io::Result<OwnedFd> {
    let raw_fd = unsafe { libc::epoll_create1(libc::EPOLL_CLOEXEC) };
    if raw_fd < 0 {
        Err(io::Error::last_os_error())
    } else {
        // SAFETY: epoll_create1 returned a fresh descriptor whose ownership
        // is transferred exactly once.
        Ok(unsafe { OwnedFd::from_raw_fd(raw_fd) })
    }
}

fn epoll_add(epoll: RawFd, descriptor: RawFd, token: u64) -> io::Result<()> {
    let mut event = libc::epoll_event {
        events: libc::EPOLLIN as u32,
        u64: token,
    };
    let result = unsafe { libc::epoll_ctl(epoll, libc::EPOLL_CTL_ADD, descriptor, &mut event) };
    if result < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

fn epoll_wait(epoll: RawFd, events: &mut [libc::epoll_event]) -> io::Result<usize> {
    loop {
        let ready = unsafe {
            libc::epoll_wait(
                epoll,
                events.as_mut_ptr(),
                i32::try_from(events.len()).unwrap_or(i32::MAX),
                -1,
            )
        };
        if ready >= 0 {
            return Ok(ready as usize);
        }
        let error = io::Error::last_os_error();
        if error.raw_os_error() != Some(libc::EINTR) {
            return Err(error);
        }
    }
}

fn sync_notifier(
    slot: &mut TransportNotifierSlot,
    runtime: &BarRuntime,
    epoll: RawFd,
) -> Result<()> {
    if let NotifierChange::Replaced { fd, .. } = slot.sync(runtime)? {
        epoll_add(epoll, fd.as_raw_fd(), SHARED_TOKEN)?;
    }
    Ok(())
}

// ============================================================================
// 4. MAIN LOOP
// ============================================================================
fn main() -> Result<()> {
    let shared_path = env::args().skip(1).last().unwrap_or_default();
    xbar_core::logging::init("xcb_wgpu_bar", &shared_path)?;

    let runtime = if shared_path.is_empty() {
        BarRuntime::new(ModelConfig::default())?
    } else {
        let recovery = TransportRecoveryConfig::new(shared_path.clone(), TRANSPORT_RETRY_INTERVAL)?;
        BarRuntime::with_managed_transport(ModelConfig::default(), recovery)?
    };

    let (conn, screen_num) = xcb::Connection::connect(None)?;
    let setup = conn.get_setup();
    let screen = setup
        .roots()
        .nth(screen_num as usize)
        .ok_or_else(|| anyhow!("no X screen found"))?;

    let presentation = PresentationConfig::default();
    let bar_height = presentation
        .bar_height
        .round()
        .clamp(1.0, f32::from(u16::MAX)) as u16;
    let font_name = env::var("XBAR_FONT").unwrap_or_else(|_| "monospace 11".to_owned());
    let font = FontDescription::from_string(&font_name);
    let mut bar = CairoBar::new(runtime, presentation, font);

    let win = conn.generate_id();
    let mut current_width = screen.width_in_pixels();
    let mut current_height = bar_height;

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

    let atoms = intern_atoms(&conn)?;
    set_dock_properties(&conn, &atoms, win, u32::from(current_width), current_height)?;

    // 绑定 WGPU
    let target = Arc::new(XcbTarget {
        conn: conn.get_raw_conn() as *mut c_void,
        screen: screen_num,
        window: win.resource_id(),
    });
    let mut gpu = pollster::block_on(Gpu::new(
        target,
        u32::from(current_width),
        u32::from(current_height),
    ))?;
    let mut cpu_frame = Vec::new();

    conn.send_and_check_request(&x::MapWindow { window: win })?;
    conn.flush()?;

    let window = WindowAdapter {
        conn: &conn,
        screen,
        atoms: &atoms,
        win,
        bar_height: Cell::new(bar_height),
        process_actions: RefCell::new(ProcessActionHandler::default()),
    };

    let mut initial_update = bar.tick();
    initial_update.merge(bar.poll_transport());
    window.apply_runtime_update(initial_update)?;

    redraw(
        &mut gpu,
        &mut cpu_frame,
        current_width,
        current_height,
        &mut bar,
    )?;

    let timer = AlignedTimer::new(Duration::from_secs(1))?;
    let epoll = create_epoll()?;
    epoll_add(epoll.as_raw_fd(), window.conn.as_raw_fd(), X_TOKEN)?;
    epoll_add(epoll.as_raw_fd(), timer.as_raw_fd(), TIMER_TOKEN)?;
    let mut notifier_slot = TransportNotifierSlot::new(true);
    sync_notifier(&mut notifier_slot, bar.runtime(), epoll.as_raw_fd())?;

    const EVENT_CAPACITY: usize = 32;
    let mut events: [libc::epoll_event; EVENT_CAPACITY] =
        std::array::from_fn(|_| libc::epoll_event { events: 0, u64: 0 });

    loop {
        let ready = epoll_wait(epoll.as_raw_fd(), &mut events)?;
        for event in events.iter().take(ready) {
            match event.u64 {
                X_TOKEN => loop {
                    let Some(x_event) = conn.poll_for_event()? else {
                        break;
                    };
                    let should_redraw = match x_event {
                        xcb::Event::X(x::Event::Expose(event)) => event.count() == 0,
                        xcb::Event::X(x::Event::ConfigureNotify(event))
                            if event.window() == win =>
                        {
                            current_width = event.width();
                            current_height = event.height();
                            window.sync_bar_height(&mut bar, event.height());
                            gpu.resize(u32::from(current_width), u32::from(current_height));
                            true
                        }
                        xcb::Event::X(x::Event::EnterNotify(event)) => bar.pointer_motion(
                            Point::new(f32::from(event.event_x()), f32::from(event.event_y())),
                        ),
                        xcb::Event::X(x::Event::MotionNotify(event)) => bar.pointer_motion(
                            Point::new(f32::from(event.event_x()), f32::from(event.event_y())),
                        ),
                        xcb::Event::X(x::Event::LeaveNotify(_)) => bar.pointer_leave(),
                        xcb::Event::X(x::Event::ButtonPress(event)) => {
                            if let Some(input) = pointer_action(event.detail()) {
                                let update = bar.pointer_action(
                                    Point::new(
                                        f32::from(event.event_x()),
                                        f32::from(event.event_y()),
                                    ),
                                    input,
                                );
                                window.apply_runtime_update(update)?
                            } else {
                                false
                            }
                        }
                        _ => false,
                    };
                    if should_redraw {
                        redraw(
                            &mut gpu,
                            &mut cpu_frame,
                            current_width,
                            current_height,
                            &mut bar,
                        )?;
                    }
                },
                TIMER_TOKEN => {
                    if timer.drain()? > 0 {
                        let mut update = bar.tick();
                        update.merge(bar.poll_transport());
                        let needs_redraw = window.apply_runtime_update(update)?;
                        sync_notifier(&mut notifier_slot, bar.runtime(), epoll.as_raw_fd())?;
                        if needs_redraw {
                            redraw(
                                &mut gpu,
                                &mut cpu_frame,
                                current_width,
                                current_height,
                                &mut bar,
                            )?;
                        }
                    }
                }
                SHARED_TOKEN => {
                    if let Some(notifier) = notifier_slot.notifier() {
                        notifier.drain()?;
                        let update = bar.poll_transport();
                        let needs_redraw = window.apply_runtime_update(update)?;
                        sync_notifier(&mut notifier_slot, bar.runtime(), epoll.as_raw_fd())?;
                        if needs_redraw {
                            redraw(
                                &mut gpu,
                                &mut cpu_frame,
                                current_width,
                                current_height,
                                &mut bar,
                            )?;
                        }
                    }
                }
                token => log::debug!("unexpected epoll token: {token}"),
            }
        }
    }
}
