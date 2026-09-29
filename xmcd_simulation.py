"""
XMCD Pump-Probe Simulation: 3D Schematic & Delay Transients
Visualizes alternating circularly polarized pump pulses and XUV probe pulses
interacting with a magnetic sample. The transmission of the probe depends on
the pump-probe delay and the pump helicity.
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Color palette: MBI Dark Navy Presentation Theme
C_BG = "#1B2436"
C_PANEL = "#222C42"
C_TEXT = "#F7F7F7"
C_MUTED = "#8E9EB2"
C_PUMP_PLUS = "#FF4D6D"   # Sigma+ Pump (Magenta)
C_PUMP_MINUS = "#FFC857"  # Sigma- Pump (Gold)
C_PROBE = "#3EC6E0"       # XUV Probe (Cyan)
C_MOUNT = "#4A5873"
C_POST = "#607290"
C_AXIS = "#3A4760"

def create_figure():
    fig = plt.figure(figsize=(16, 9), facecolor=C_BG)
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[1.60, 1.05],
        height_ratios=[1.0, 1.0],
        left=0.03, right=0.96, top=0.94, bottom=0.09,
        wspace=0.22, hspace=0.28
    )

    ax_3d = fig.add_subplot(gs[:, 0], projection='3d', facecolor=C_BG)
    ax_t = fig.add_subplot(gs[0, 1], facecolor=C_PANEL)
    ax_asym = fig.add_subplot(gs[1, 1], facecolor=C_PANEL)

    return fig, ax_3d, ax_t, ax_asym

def setup_3d_axes(ax_3d):
    ax_3d.set_box_aspect((3.8, 1.2, 1.2))
    ax_3d.view_init(elev=18, azim=-62)

    ax_3d.xaxis.set_pane_color((0, 0, 0, 0))
    ax_3d.yaxis.set_pane_color((0, 0, 0, 0))
    ax_3d.zaxis.set_pane_color((0, 0, 0, 0))
    ax_3d.grid(False)
    ax_3d.set_axis_off()

    ax_3d.set_xlim(-0.2, 7.2)
    ax_3d.set_ylim(-1.6, 1.6)
    ax_3d.set_zlim(-1.7, 1.7)

    z_rail = np.linspace(-0.2, 7.0, 100)
    ax_3d.plot(z_rail, np.zeros_like(z_rail), np.zeros_like(z_rail),
               color=C_AXIS, linestyle='--', linewidth=1.5, zorder=1)
    ax_3d.plot(z_rail, np.zeros_like(z_rail), np.full_like(z_rail, -1.5),
               color=C_MOUNT, linestyle='-', linewidth=3.0, alpha=0.5, zorder=1)

def draw_sample(ax_3d, z_pos, label):
    """Draws a magnetic sample."""
    R = 1.0
    theta = np.linspace(0, 2 * np.pi, 60)

    # Post
    ax_3d.plot([z_pos, z_pos], [0, 0], [-R, -1.5], color=C_POST, linewidth=4.5, solid_capstyle='round')
    ax_3d.plot([z_pos - 0.15, z_pos + 0.15], [0, 0], [-1.5, -1.5], color=C_MOUNT, linewidth=6.0, solid_capstyle='round')

    # Sample Disk
    r = np.linspace(0, R, 15)
    TH, RR = np.meshgrid(theta, r)
    disc_x = RR * np.cos(TH)
    disc_y = RR * np.sin(TH)
    disc_z = np.full_like(disc_x, z_pos)
    ax_3d.plot_surface(disc_z, disc_x, disc_y, alpha=0.7, color='#7A89A3', edgecolor='none')

    # Magnetization arrow (M)
    ax_3d.quiver(z_pos, 0, 0, 0.6, 0, 0, color='white', linewidth=3, arrow_length_ratio=0.3)
    ax_3d.text(z_pos+0.7, 0, 0.2, "M", color='white', fontsize=12, fontweight='bold')

    ax_3d.text(z_pos, 0, R + 0.32, label, color=C_TEXT, fontsize=14, fontweight='bold', ha='center')

def setup_2d_plots(ax_t, ax_asym, delays):
    for ax in (ax_t, ax_asym):
        ax.set_facecolor(C_PANEL)
        for spine in ax.spines.values():
            spine.set_color(C_MUTED)
            spine.set_alpha(0.4)
        ax.tick_params(colors=C_MUTED, labelsize=9.5)
        ax.set_xlim(delays[0], delays[-1])
        ax.axhline(0, color=C_MUTED, linestyle=':', linewidth=0.9, alpha=0.6)
        ax.grid(True, linestyle=':', alpha=0.25, color=C_MUTED)

    # Top: Transmission
    ax_t.set_xticklabels([])
    ax_t.set_ylabel(r"XUV Transmission ($T^+$ and $T^-$)", color=C_TEXT, fontsize=11, fontweight='bold')
    ax_t.set_ylim(0.2, 1.1)

    # Pre-draw background transients
    T0 = 0.8
    A_max = 0.4
    tau_relax = 2.0

    T_plus = np.where(delays < 0, T0, T0 - A_max * np.exp(-delays/tau_relax))
    T_minus = np.where(delays < 0, T0, T0 - 0.3 * A_max * np.exp(-delays/tau_relax))

    ax_t.plot(delays, T_plus, color=C_PUMP_PLUS, linewidth=2, alpha=0.4)
    ax_t.plot(delays, T_minus, color=C_PUMP_MINUS, linewidth=2, alpha=0.4)

    # Dynamic lines
    line_t_plus, = ax_t.plot([], [], color=C_PUMP_PLUS, linewidth=2.5)
    line_t_minus, = ax_t.plot([], [], color=C_PUMP_MINUS, linewidth=2.5)

    marker_t_plus, = ax_t.plot([], [], 'o', color=C_PUMP_PLUS, markersize=8)
    marker_t_minus, = ax_t.plot([], [], 'o', color=C_PUMP_MINUS, markersize=8)
    vline_t = ax_t.axvline(delays[0], color='white', linestyle='--', alpha=0.5)

    # Bottom: Asymmetry
    ax_asym.set_xlabel(r"Pump-Probe Delay $\tau$ (ps)", color=C_TEXT, fontsize=11, fontweight='bold')
    ax_asym.set_ylabel(r"XMCD Asymmetry $\Delta T$", color=C_TEXT, fontsize=11, fontweight='bold')
    ax_asym.set_ylim(-0.1, max(np.max(T_minus - T_plus)*1.2, 0.1))

    Asym = T_minus - T_plus
    ax_asym.plot(delays, Asym, color='white', linewidth=2, alpha=0.4)

    line_asym, = ax_asym.plot([], [], color='white', linewidth=2.5)
    marker_asym, = ax_asym.plot([], [], 'o', color='white', markersize=8)
    vline_asym = ax_asym.axvline(delays[0], color='white', linestyle='--', alpha=0.5)

    return line_t_plus, line_t_minus, marker_t_plus, marker_t_minus, vline_t, line_asym, marker_asym, vline_asym, T_plus, T_minus, Asym

def build_simulation(output_path=None, preview_only=False, fps=30, duration_sec=12.0):
    fig, ax_3d, ax_t, ax_asym = create_figure()
    setup_3d_axes(ax_3d)

    z_sample = 3.5
    z_end = 7.0
    draw_sample(ax_3d, z_sample, "Magnetic Sample")

    # Time axes
    delays = np.linspace(-2.0, 6.0, 300)
    (line_t_plus, line_t_minus, marker_t_plus, marker_t_minus, vline_t,
     line_asym, marker_asym, vline_asym, T_plus_curve, T_minus_curve, Asym_curve) = setup_2d_plots(ax_t, ax_asym, delays)

    # 3D lines
    line_pump, = ax_3d.plot([], [], [], linewidth=2.5)
    line_probe_in, = ax_3d.plot([], [], [], color=C_PROBE, linewidth=2.0)
    line_probe_out, = ax_3d.plot([], [], [], color=C_PROBE, linewidth=2.0)

    total_frames = int(fps * duration_sec)

    # Fast time vs slow time
    # Fast time 't_fast' moves the pulses through space.
    # Slow time 'tau' is the delay (offset between pump and probe).
    # We will loop through 'tau' over the animation.

    # Pulse params
    k_pump = 2*np.pi / 0.8
    k_probe = 2*np.pi / 0.3
    sigma_z = 0.4
    v = 8.0 # speed of light in simulation

    pts_in = np.linspace(0, z_sample, 150)
    pts_out = np.linspace(z_sample, z_end, 150)

    def update(frame_idx):
        progress = frame_idx / float(total_frames)
        # Slow delay tau sweeps from delays[0] to delays[-1]
        current_tau = delays[0] + progress * (delays[-1] - delays[0])

        # Fast time oscillates to move pulses
        # Let's say one full fast cycle per 1 second of animation
        t_fast = (frame_idx % fps) / float(fps) * (z_end / v * 1.5)

        # Determine pump helicity for this cycle
        cycle_idx = frame_idx // fps
        is_sigma_plus = (cycle_idx % 2 == 0)

        pump_color = C_PUMP_PLUS if is_sigma_plus else C_PUMP_MINUS
        line_pump.set_color(pump_color)

        # Pump position
        z_c_pump = v * t_fast
        env_pump = np.exp(-0.5 * ((pts_in - z_c_pump) / sigma_z)**2)
        phase_pump = k_pump * (pts_in - z_c_pump)

        if is_sigma_plus:
            Ex_pump = env_pump * np.cos(phase_pump)
            Ey_pump = env_pump * np.sin(phase_pump)
        else:
            Ex_pump = env_pump * np.cos(phase_pump)
            Ey_pump = -env_pump * np.sin(phase_pump)

        # Only show pump before sample
        mask_pump = pts_in <= z_sample
        line_pump.set_data(Ex_pump[mask_pump], Ey_pump[mask_pump])
        line_pump.set_3d_properties(pts_in[mask_pump], zdir='x')

        # Probe position (delayed by tau)
        # In spatial terms, delay tau means the probe is behind the pump by v*tau
        z_c_probe = v * t_fast - current_tau

        # Probe IN
        env_probe_in = np.exp(-0.5 * ((pts_in - z_c_probe) / (sigma_z*0.5))**2)
        phase_probe_in = k_probe * (pts_in - z_c_probe)
        Ex_probe_in = env_probe_in * np.cos(phase_probe_in)
        Ey_probe_in = np.zeros_like(pts_in)

        mask_probe_in = pts_in <= z_sample
        line_probe_in.set_data(Ex_probe_in[mask_probe_in], Ey_probe_in[mask_probe_in])
        line_probe_in.set_3d_properties(pts_in[mask_probe_in], zdir='x')

        # Probe OUT (transmitted)
        env_probe_out = np.exp(-0.5 * ((pts_out - z_c_probe) / (sigma_z*0.5))**2)
        phase_probe_out = k_probe * (pts_out - z_c_probe)

        # Calculate current transmission based on current_tau
        # Interpolate
        idx_tau = np.argmin(np.abs(delays - current_tau))
        T_curr = T_plus_curve[idx_tau] if is_sigma_plus else T_minus_curve[idx_tau]

        Ex_probe_out = env_probe_out * T_curr * np.cos(phase_probe_out)
        Ey_probe_out = np.zeros_like(pts_out)

        mask_probe_out = pts_out > z_sample
        line_probe_out.set_data(Ex_probe_out[mask_probe_out], Ey_probe_out[mask_probe_out])
        line_probe_out.set_3d_properties(pts_out[mask_probe_out], zdir='x')

        # Update 2D graphs
        line_t_plus.set_data(delays[:idx_tau], T_plus_curve[:idx_tau])
        line_t_minus.set_data(delays[:idx_tau], T_minus_curve[:idx_tau])
        line_asym.set_data(delays[:idx_tau], Asym_curve[:idx_tau])

        vline_t.set_xdata([current_tau, current_tau])
        vline_asym.set_xdata([current_tau, current_tau])

        marker_t_plus.set_data([current_tau], [T_plus_curve[idx_tau]])
        marker_t_minus.set_data([current_tau], [T_minus_curve[idx_tau]])
        marker_asym.set_data([current_tau], [Asym_curve[idx_tau]])

        return (line_pump, line_probe_in, line_probe_out,
                line_t_plus, line_t_minus, line_asym,
                vline_t, vline_asym, marker_t_plus, marker_t_minus, marker_asym)

    if preview_only:
        update(int(total_frames * 0.4))
        preview_file = Path("xmcd_preview.png")
        fig.savefig(preview_file, dpi=120, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        print(f"Generated preview image: {preview_file.resolve()}")
        return

    if output_path is not None:
        out_p = Path(output_path)
        print(f"Exporting animation to {out_p.name} ({total_frames} frames @ {fps} fps)...")
        ani = FuncAnimation(fig, update, frames=total_frames, interval=1000.0 / fps, blit=False)

        if out_p.suffix.lower() == ".mp4":
            writer = FFMpegWriter(fps=fps, metadata=dict(artist='Max Dittmer'), bitrate=4500)
            ani.save(out_p, writer=writer, dpi=120)
        elif out_p.suffix.lower() == ".gif":
            writer = PillowWriter(fps=fps)
            ani.save(out_p, writer=writer)
        plt.close(fig)
        print(f"Successfully saved animation to {out_p.resolve()}")
        return

    ani = FuncAnimation(fig, update, frames=total_frames, interval=1000.0 / fps, blit=False)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XMCD Pump-Probe Simulation")
    parser.add_argument("--preview", action="store_true", help="Save a single high-res preview frame")
    parser.add_argument("--output", type=str, default=None, help="Output filename (.mp4 or .gif)")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--duration", type=float, default=12.0, help="Duration of full cycle in seconds")
    args = parser.parse_args()

    build_simulation(
        output_path=args.output,
        preview_only=args.preview,
        fps=args.fps,
        duration_sec=args.duration
    )
