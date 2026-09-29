"""
XMCD Pump-Probe Simulation: 3D Schematic & Delay Transients
Visualizes alternating circularly polarized pump pulses (sigma+ / sigma-)
and circularly polarized delayed XUV probe pulses interacting with a magnetic sample.
Features exaggerated amplitude absorption contrast (constant linewidth)
and calm, slow-motion pulse propagation.
"""
import sys
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
    ax_3d.set_zlim(-1.7, 2.2)

    z_rail = np.linspace(-0.2, 7.0, 100)
    ax_3d.plot(z_rail, np.zeros_like(z_rail), np.zeros_like(z_rail),
               color=C_AXIS, linestyle='--', linewidth=1.5, zorder=1)
    ax_3d.plot(z_rail, np.zeros_like(z_rail), np.full_like(z_rail, -1.5),
               color=C_MOUNT, linestyle='-', linewidth=3.0, alpha=0.5, zorder=1)

def draw_sample(ax_3d, z_pos):
    """Draws magnetic sample disc with clamp and magnetization arrow."""
    R = 1.05
    theta = np.linspace(0, 2 * np.pi, 60)

    # Optical Post
    ax_3d.plot([z_pos, z_pos], [0, 0], [-R, -1.5], color=C_POST, linewidth=4.5, solid_capstyle='round')
    ax_3d.plot([z_pos - 0.15, z_pos + 0.15], [0, 0], [-1.5, -1.5], color=C_MOUNT, linewidth=6.0, solid_capstyle='round')

    # Mount Ring
    ring_x = R * np.cos(theta)
    ring_y = R * np.sin(theta)
    ring_z = np.full_like(ring_x, z_pos)
    ax_3d.plot(ring_z, ring_x, ring_y, color=C_MOUNT, linewidth=3.0)

    # Sample Disc Surface
    r = np.linspace(0, R * 0.92, 15)
    TH, RR = np.meshgrid(theta, r)
    disc_x = RR * np.cos(TH)
    disc_y = RR * np.sin(TH)
    disc_z = np.full_like(disc_x, z_pos)
    ax_3d.plot_surface(disc_z, disc_x, disc_y, alpha=0.45, color='#5E6E88', edgecolor='none')

    # Magnetization vector arrow (M along optical axis)
    ax_3d.quiver(z_pos, 0, 0, 0.75, 0, 0, color='white', linewidth=3.2, arrow_length_ratio=0.28)
    ax_3d.text(z_pos + 0.85, 0, 0.18, r"$\mathbf{M}$", color='white', fontsize=13, fontweight='bold')

    # Minimal label above sample
    ax_3d.text(z_pos, 0, R + 0.28, "Magnetic Sample (Pt/Co)", color=C_TEXT, fontsize=12.5, fontweight='bold', ha='center')

def draw_detector(ax_3d, z_pos):
    """Draws detection screen at the output."""
    S = 1.15
    y_screen = [-S, S, S, -S]
    z_screen = [S, S, -S, -S]
    x_screen = [z_pos, z_pos, z_pos, z_pos]
    verts = [list(zip(x_screen, y_screen, z_screen))]
    poly = Poly3DCollection(verts, alpha=0.15, facecolor=C_PANEL, edgecolor=C_MUTED, linewidth=1.2)
    ax_3d.add_collection3d(poly)
    ax_3d.text(z_pos, 0, 1.45, "XUV Spectrometer", color=C_PROBE, fontsize=11, fontweight='bold', ha='center')

def setup_2d_plots(ax_t, ax_asym, delays):
    tau_relax = 1.8

    # Exaggerated transmission curves:
    # Baseline T0 = 1.0 (before excitation)
    # Sigma+ pump -> heavy absorption -> T+ drops to 0.08 (92% drop!)
    # Sigma- pump -> weak absorption  -> T- drops to 0.82 (18% drop!)
    T0 = 1.0
    T_plus_curve = np.where(delays < 0, T0, T0 - 0.92 * np.exp(-delays / tau_relax))
    T_minus_curve = np.where(delays < 0, T0, T0 - 0.18 * np.exp(-delays / tau_relax))
    Asym_curve = T_minus_curve - T_plus_curve

    for ax in (ax_t, ax_asym):
        ax.set_facecolor(C_PANEL)
        for spine in ax.spines.values():
            spine.set_color(C_MUTED)
            spine.set_alpha(0.4)
        ax.tick_params(colors=C_MUTED, labelsize=9.5)
        ax.set_xlim(delays[0], delays[-1])
        ax.grid(True, linestyle=':', alpha=0.25, color=C_MUTED)

    # Top Plot: XUV Transmission
    ax_t.set_xticklabels([])
    ax_t.set_ylabel(r"XUV Transmission ($T^+$ and $T^-$)", color=C_TEXT, fontsize=11, fontweight='bold')
    ax_t.set_ylim(0.0, 1.15)
    ax_t.set_yticks([0.0, 0.5, 1.0])
    ax_t.set_yticklabels(["0.0 (Absorbed)", "0.5", "1.0 (Full)"])
    ax_t.axhline(1.0, color=C_MUTED, linestyle=':', linewidth=0.8, alpha=0.4)

    # Background theoretical reference traces
    ax_t.plot(delays, T_plus_curve, color=C_PUMP_PLUS, linewidth=2.0, alpha=0.35, label=r"$\sigma^+$ Pump")
    ax_t.plot(delays, T_minus_curve, color=C_PUMP_MINUS, linewidth=2.0, alpha=0.35, label=r"$\sigma^-$ Pump")
    ax_t.legend(loc='lower right', facecolor=C_PANEL, edgecolor='none', labelcolor=C_TEXT, fontsize=9.5)

    # Dynamic line traces and tracking markers
    line_t_plus, = ax_t.plot([], [], color=C_PUMP_PLUS, linewidth=2.6)
    line_t_minus, = ax_t.plot([], [], color=C_PUMP_MINUS, linewidth=2.6)
    marker_t_plus, = ax_t.plot([], [], 'o', color=C_PUMP_PLUS, markersize=8.5, markeredgecolor='white', markeredgewidth=1.2)
    marker_t_minus, = ax_t.plot([], [], 'o', color=C_PUMP_MINUS, markersize=8.5, markeredgecolor='white', markeredgewidth=1.2)
    vline_t = ax_t.axvline(delays[0], color='white', linestyle='--', linewidth=1.0, alpha=0.5)

    # Bottom Plot: XMCD Asymmetry Delta T
    ax_asym.set_xlabel(r"Pump-Probe Delay $\tau$ (ps)", color=C_TEXT, fontsize=11, fontweight='bold')
    ax_asym.set_ylabel(r"XMCD Contrast $\Delta T = T^- - T^+$", color=C_TEXT, fontsize=11, fontweight='bold')
    ax_asym.set_ylim(-0.05, 0.90)
    ax_asym.set_yticks([0.0, 0.35, 0.74])
    ax_asym.axhline(0.0, color=C_MUTED, linestyle=':', linewidth=0.9, alpha=0.6)

    ax_asym.plot(delays, Asym_curve, color='white', linewidth=2.0, alpha=0.35)
    line_asym, = ax_asym.plot([], [], color='white', linewidth=2.6)
    marker_asym, = ax_asym.plot([], [], 'o', color='white', markersize=8.5, markeredgecolor=C_PROBE, markeredgewidth=1.4)
    vline_asym = ax_asym.axvline(delays[0], color='white', linestyle='--', linewidth=1.0, alpha=0.5)

    return (
        line_t_plus, line_t_minus, marker_t_plus, marker_t_minus, vline_t,
        line_asym, marker_asym, vline_asym,
        T_plus_curve, T_minus_curve, Asym_curve
    )

def build_simulation(output_path=None, preview_only=False, fps=30, duration_sec=14.0, pulse_speed=1.35):
    fig, ax_3d, ax_t, ax_asym = create_figure()
    setup_3d_axes(ax_3d)

    z_sample = 3.5
    z_screen = 6.6
    draw_sample(ax_3d, z_sample)
    draw_detector(ax_3d, z_screen)

    # Informational status badge in 3D (placed cleanly above sample mount)
    badge_text = ax_3d.text(3.5, 0.0, 1.95, "", fontsize=12.5, fontweight='bold', ha='center', color=C_PUMP_PLUS)

    # Delay range
    delays = np.linspace(-1.5, 5.0, 300)
    (
        line_t_plus, line_t_minus, marker_t_plus, marker_t_minus, vline_t,
        line_asym, marker_asym, vline_asym,
        T_plus_curve, T_minus_curve, Asym_curve
    ) = setup_2d_plots(ax_t, ax_asym, delays)

    # 3D pulse wave objects
    # Pump pulse: large circular pulse
    line_pump, = ax_3d.plot([], [], [], linewidth=3.0)
    # Stream of smaller probe pulses: constant linewidth (2.4), tight circular helices
    line_probe, = ax_3d.plot([], [], [], color=C_PROBE, linewidth=2.4)

    # Discretized coordinates along beamline
    pts_axis = np.linspace(-0.2, z_screen, 460)

    # Physics parameters:
    # Pump: optical carrier wavelength (800 nm equivalent)
    lambda_pump = 0.50
    k_pump = 2 * np.pi / lambda_pump
    sigma_pump = 0.30
    A_pump = 0.85

    # Probe: circular XUV probe pulse with shorter carrier wavelength (tighter corkscrew)
    lambda_probe = 0.22
    k_probe = 2 * np.pi / lambda_probe
    sigma_probe = 0.15
    A_probe_in = 0.48
    d_probe = 0.80             # Spacing between consecutive probe pulses in stream
    v_probe = 1.65             # Fast stream of smaller probe pulses

    # Pump timing & linear trajectory across beamline
    total_frames = int(fps * duration_sec)
    half_frames = total_frames // 2
    half_period = duration_sec / 2.0

    z_pump_start = 0.8
    z_pump_end = 6.4
    v_pump = (z_pump_end - z_pump_start) / half_period
    t_hit = (z_sample - z_pump_start) / v_pump   # exact time when pump hits sample

    def update(frame_idx):
        # Two equal phases:
        # Phase 1: Sigma+ pump (0 to half_frames)
        # Phase 2: Sigma- pump (half_frames to total_frames)
        is_sigma_plus = (frame_idx < half_frames)
        cycle_frame = frame_idx if is_sigma_plus else (frame_idx - half_frames)
        progress = cycle_frame / float(half_frames)

        # Current time within cycle
        t_cycle = progress * half_period

        # 1. Pump position and synchronous delay parameter tau
        z_pump = z_pump_start + v_pump * t_cycle

        if t_cycle < t_hit:
            current_tau = delays[0] * (1.0 - t_cycle / t_hit)   # -1.5 ps -> 0.0 ps
        else:
            current_tau = delays[-1] * ((t_cycle - t_hit) / (half_period - t_hit)) # 0.0 ps -> 5.0 ps

        idx_tau = int(np.clip(np.searchsorted(delays, current_tau), 0, len(delays) - 1))

        # Update status badge in 3D
        if t_cycle < t_hit - 0.4:
            badge_text.set_text(r"Pump Approaching Sample ($\tau < 0$, Ground State)")
            badge_text.set_color(C_MUTED)
        elif is_sigma_plus:
            badge_text.set_text(r"Pump: $\sigma^+$ (Circular)  $\rightarrow$  Heavy Probe Absorption ($T^+ = 8\%$)")
            badge_text.set_color(C_PUMP_PLUS)
        else:
            badge_text.set_text(r"Pump: $\sigma^-$ (Circular)  $\rightarrow$  High Probe Transmission ($T^- = 82\%$)")
            badge_text.set_color(C_PUMP_MINUS)

        # 2. Render Pump Pulse (Single prominent circular wave packet)
        pump_color = C_PUMP_PLUS if is_sigma_plus else C_PUMP_MINUS
        line_pump.set_color(pump_color)

        # Pump passes through with transmitted intensity after sample
        A_pump_curr = np.where(pts_axis < z_sample, A_pump, A_pump * 0.55)
        env_pump = np.exp(-0.5 * ((pts_axis - z_pump) / sigma_pump)**2)
        phase_pump = k_pump * (pts_axis - z_pump)
        mask_pump = env_pump > 0.015

        if is_sigma_plus:
            Ex_pump = np.where(mask_pump, env_pump * A_pump_curr * np.cos(phase_pump), np.nan)
            Ey_pump = np.where(mask_pump, env_pump * A_pump_curr * np.sin(phase_pump), np.nan)
        else:
            Ex_pump = np.where(mask_pump, env_pump * A_pump_curr * np.cos(phase_pump), np.nan)
            Ey_pump = np.where(mask_pump, -env_pump * A_pump_curr * np.sin(phase_pump), np.nan)

        line_pump.set_data(Ex_pump, Ey_pump)
        line_pump.set_3d_properties(pts_axis, zdir='x')

        # 3. Render Stream of Smaller Probe Pulses
        # Continuous pulse train flowing along optical axis
        Ex_probe = np.full_like(pts_axis, np.nan)
        Ey_probe = np.full_like(pts_axis, np.nan)

        # Range of packet indices currently crossing or near the rail
        k_min = int(np.floor((v_probe * t_cycle - z_screen - 1.0) / d_probe))
        k_max = int(np.ceil((v_probe * t_cycle - (-0.5) + 1.0) / d_probe))

        for k in range(k_min, k_max + 1):
            z_pk = v_probe * t_cycle - k * d_probe
            if z_pk < -0.6 or z_pk > z_screen + 0.6:
                continue

            # Moment this packet crossed the sample disc (z = z_sample)
            t_cross_k = (z_sample + k * d_probe) / v_probe

            if t_cross_k < t_hit:
                # Crossed before pump: ground state transmission
                T_k = 1.0
            else:
                # Crossed after pump: sample is demagnetized and relaxing!
                tau_k = delays[-1] * ((t_cross_k - t_hit) / (half_period - t_hit))
                tau_relax = 1.8
                if is_sigma_plus:
                    T_k = 1.0 - 0.92 * np.exp(-max(0.0, tau_k) / tau_relax)
                else:
                    T_k = 1.0 - 0.18 * np.exp(-max(0.0, tau_k) / tau_relax)

            # Local envelope for packet k
            mask_k = np.abs(pts_axis - z_pk) < 3.2 * sigma_probe
            if not np.any(mask_k):
                continue

            z_sub = pts_axis[mask_k]
            env_k = np.exp(-0.5 * ((z_sub - z_pk) / sigma_probe)**2)
            phase_k = k_probe * (z_sub - z_pk)

            # Amplitude: A_probe_in before sample, attenuated by T_k after sample
            A_eff = np.where(z_sub < z_sample, A_probe_in, A_probe_in * T_k)

            valid = env_k > 0.015
            Ex_sub = np.where(valid, env_k * A_eff * np.cos(phase_k), np.nan)
            Ey_sub = np.where(valid, env_k * A_eff * np.sin(phase_k), np.nan)

            # Combine into overall probe line
            Ex_probe[mask_k] = np.where(valid, Ex_sub, Ex_probe[mask_k])
            Ey_probe[mask_k] = np.where(valid, Ey_sub, Ey_probe[mask_k])

        line_probe.set_data(Ex_probe, Ey_probe)
        line_probe.set_3d_properties(pts_axis, zdir='x')

        # 4. Update 2D Graphs
        if is_sigma_plus:
            line_t_plus.set_data(delays[:idx_tau], T_plus_curve[:idx_tau])
            line_t_minus.set_data([], [])
            marker_t_plus.set_data([current_tau], [T_plus_curve[idx_tau]])
            marker_t_minus.set_data([], [])
        else:
            line_t_plus.set_data(delays, T_plus_curve)
            line_t_minus.set_data(delays[:idx_tau], T_minus_curve[:idx_tau])
            marker_t_plus.set_data([], [])
            marker_t_minus.set_data([current_tau], [T_minus_curve[idx_tau]])

        line_asym.set_data(delays[:idx_tau], Asym_curve[:idx_tau])
        marker_asym.set_data([current_tau], [Asym_curve[idx_tau]])

        vline_t.set_xdata([current_tau, current_tau])
        vline_asym.set_xdata([current_tau, current_tau])

        return (
            line_pump, line_probe, badge_text,
            line_t_plus, line_t_minus, line_asym,
            marker_t_plus, marker_t_minus, marker_asym,
            vline_t, vline_asym
        )

    if preview_only:
        update(int(half_frames * 0.60))
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
    parser = argparse.ArgumentParser(description="XMCD Pump-Probe Simulation with Circular Probe & Exaggerated Amplitude")
    parser.add_argument("--preview", action="store_true", help="Save a single high-res preview frame")
    parser.add_argument("--output", type=str, default=None, help="Output filename (.mp4 or .gif)")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--duration", type=float, default=20.0, help="Duration of full two-pulse cycle in seconds")
    parser.add_argument("--speed", type=float, default=1.35, help="Pulse propagation speed along optical axis")
    args = parser.parse_args()

    build_simulation(
        output_path=args.output,
        preview_only=args.preview,
        fps=args.fps,
        duration_sec=args.duration,
        pulse_speed=args.speed
    )
