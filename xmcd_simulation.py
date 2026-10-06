"""
XMCD Time-Resolved Pump-Probe 3D Simulation
Author: Max Dittmer
Styling: MBI Corporate Identity (Dark Navy / Modern Card UI)

Displays dual 3D optical beamlines:
- TOP 3D:    sigma+ pump excitation -> Heavy probe absorption (T+ = 8%)
- BOTTOM 3D: sigma- pump excitation -> High probe transmission (T- = 82%)
- RIGHT 2D:  Top: Transmissions T+ and T- vs Delay tau
             Bottom: XMCD Contrast Delta T = T- - T+ vs Delay tau

Features:
- Pump pulse is 100% absorbed at the sample disc (does not transmit through!).
- Both polarizations run SIMULTANEOUSLY for direct side-by-side comparison.
- Stream of compact 3D circular probe helices continuously passes through.
- Constant linewidth across all pulses.
- Slow, deliberate animation speed for clear presentation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from pathlib import Path
import argparse

# -----------------------------------------------------------------------------
# Color Palette (Strictly aligned with MBI Beamer Theme)
# -----------------------------------------------------------------------------
C_BG         = '#1B2436'    # Main presentation background
C_PANEL      = '#222C42'    # Plot card background
C_TEXT       = '#E8EDF5'    # Primary typography
C_MUTED      = '#8B9BB4'    # Subtitles, axes, grids
C_PUMP_PLUS  = '#FF4D6D'    # Sigma+ pump: vibrant magenta/pink
C_PUMP_MINUS = '#FFC857'    # Sigma- pump: rich amber gold
C_PROBE      = '#3EC6E0'    # XUV probe: cyan helix
C_MOUNT      = '#3A4863'    # Optics mount hardware
C_POST       = '#2A364F'    # Optical post
C_AXIS       = '#4A5A78'    # Optical axis dashed rail

def create_figure():
    fig = plt.figure(figsize=(16, 9), facecolor=C_BG)
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[2.20, 1.0],
        height_ratios=[1.0, 1.0],
        left=0.00, right=0.97, top=0.96, bottom=0.06,
        wspace=0.08, hspace=0.04
    )

    ax_3d_plus  = fig.add_subplot(gs[0, 0], projection='3d', facecolor=C_BG)
    ax_3d_minus = fig.add_subplot(gs[1, 0], projection='3d', facecolor=C_BG)
    ax_t        = fig.add_subplot(gs[0, 1], facecolor=C_PANEL)
    ax_asym     = fig.add_subplot(gs[1, 1], facecolor=C_PANEL)

    return fig, ax_3d_plus, ax_3d_minus, ax_t, ax_asym

def setup_3d_axes(ax_3d, title_text, title_color):
    # Zoom in camera massively to fill the entire subplot rectangle
    ax_3d.set_box_aspect((4.4, 1.1, 0.95), zoom=1.65)
    ax_3d.view_init(elev=14, azim=-62)

    ax_3d.xaxis.set_pane_color((0, 0, 0, 0))
    ax_3d.yaxis.set_pane_color((0, 0, 0, 0))
    ax_3d.zaxis.set_pane_color((0, 0, 0, 0))
    ax_3d.grid(False)
    ax_3d.set_axis_off()

    ax_3d.set_xlim(0.0, 7.0)
    ax_3d.set_ylim(-1.20, 1.20)
    ax_3d.set_zlim(-1.42, 1.25)

    z_rail = np.linspace(-0.2, 7.0, 100)
    ax_3d.plot(z_rail, np.zeros_like(z_rail), np.zeros_like(z_rail),
               color=C_AXIS, linestyle='--', linewidth=1.2, zorder=1)
    ax_3d.plot(z_rail, np.zeros_like(z_rail), np.full_like(z_rail, -1.4),
               color=C_MOUNT, linestyle='-', linewidth=2.8, alpha=0.5, zorder=1)

    # Clean 2D screen-space title badge filling upper region
    badge = ax_3d.text2D(0.48, 0.93, title_text, transform=ax_3d.transAxes,
                         fontsize=13.5, fontweight='bold', ha='center', color=title_color)
    return badge

def draw_sample(ax_3d, z_pos):
    """Draws sample disc with clamp (no in-plane M arrow — magnetization is out of plane)."""
    R = 1.02
    theta = np.linspace(0, 2 * np.pi, 60)

    # Optical Post
    ax_3d.plot([z_pos, z_pos], [0, 0], [-R, -1.4], color=C_POST, linewidth=4.0, solid_capstyle='round')
    ax_3d.plot([z_pos - 0.15, z_pos + 0.15], [0, 0], [-1.4, -1.4], color=C_MOUNT, linewidth=5.5, solid_capstyle='round')

    # Mount Ring
    ring_x = R * np.cos(theta)
    ring_y = R * np.sin(theta)
    ring_z = np.full_like(ring_x, z_pos)
    ax_3d.plot(ring_z, ring_x, ring_y, color=C_MOUNT, linewidth=2.6)

    # Sample Disc Surface
    r = np.linspace(0, R * 0.92, 14)
    TH, RR = np.meshgrid(theta, r)
    disc_x = RR * np.cos(TH)
    disc_y = RR * np.sin(TH)
    disc_z = np.full_like(disc_x, z_pos)
    ax_3d.plot_surface(disc_z, disc_x, disc_y, alpha=0.45, color='#5E6E88', edgecolor='none')

def draw_detector(ax_3d, z_pos):
    """Draws detection screen at the output."""
    S = 1.10
    y_screen = [-S, S, S, -S]
    z_screen = [S, S, -S, -S]
    x_screen = [z_pos, z_pos, z_pos, z_pos]
    verts = [list(zip(x_screen, y_screen, z_screen))]
    poly = Poly3DCollection(verts, alpha=0.15, facecolor=C_PANEL, edgecolor=C_MUTED, linewidth=1.1)
    ax_3d.add_collection3d(poly)
    ax_3d.text(z_pos, 0, 1.35, "XUV Spectrometer", color=C_PROBE, fontsize=9.5, fontweight='bold', ha='center')

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
    ax_t.set_yticklabels(["0.0 (Absorbed)", "0.5", "1.0 (Transmitted)"])
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

def build_simulation(output_path=None, preview_only=False, fps=30, duration_sec=16.0):
    fig, ax_3d_plus, ax_3d_minus, ax_t, ax_asym = create_figure()

    z_sample = 3.5
    z_screen = 6.6

    # Setup 3D environments
    draw_sample(ax_3d_plus, z_sample)
    draw_detector(ax_3d_plus, z_screen)
    badge_plus = setup_3d_axes(
        ax_3d_plus,
        r"$\mathbf{\sigma^+}$ Pump",
        C_PUMP_PLUS
    )

    draw_sample(ax_3d_minus, z_sample)
    draw_detector(ax_3d_minus, z_screen)
    badge_minus = setup_3d_axes(
        ax_3d_minus,
        r"$\mathbf{\sigma^-}$ Pump",
        C_PUMP_MINUS
    )

    # Delay range
    delays = np.linspace(-1.5, 5.0, 300)
    (
        line_t_plus, line_t_minus, marker_t_plus, marker_t_minus, vline_t,
        line_asym, marker_asym, vline_asym,
        T_plus_curve, T_minus_curve, Asym_curve
    ) = setup_2d_plots(ax_t, ax_asym, delays)

    # 3D pulse wave objects (Constant linewidth!)
    # Top setup: sigma+
    line_pump_plus,  = ax_3d_plus.plot([], [], [], color=C_PUMP_PLUS, linewidth=2.8)
    line_probe_plus, = ax_3d_plus.plot([], [], [], color=C_PROBE, linewidth=2.4)

    # Bottom setup: sigma-
    line_pump_minus,  = ax_3d_minus.plot([], [], [], color=C_PUMP_MINUS, linewidth=2.8)
    line_probe_minus, = ax_3d_minus.plot([], [], [], color=C_PROBE, linewidth=2.4)

    # Discretization grids:
    # Pump pulse exists ONLY BEFORE SAMPLE (completely absorbed at sample!)
    pts_pump = np.linspace(-0.2, z_sample, 180)
    # Probe pulses exist across entire beamline through sample to detector
    pts_probe = np.linspace(-0.2, z_screen, 460)

    # Physics parameters:
    # Pump: optical carrier wavelength (800 nm equivalent)
    lambda_pump = 0.50
    k_pump = 2 * np.pi / lambda_pump
    sigma_pump = 0.28
    A_pump = 0.82

    # Probe: circular XUV probe pulse with shorter carrier wavelength (tighter corkscrew)
    lambda_probe = 0.22
    k_probe = 2 * np.pi / lambda_probe
    sigma_probe = 0.15
    A_probe_in = 0.46
    d_probe = 0.82             # Spacing between consecutive probe pulses in stream
    # Scale with duration so a shorter cycle stays spatially equivalent (+20% wall-clock speed).
    v_probe = 1.60 * (16.0 / duration_sec)

    # Pump timing & linear trajectory towards sample
    total_frames = int(fps * duration_sec)
    z_pump_start = 0.5
    t_hit = 4.2 * (duration_sec / 16.0)  # pump hits sample at same relative progress
    v_pump = (z_sample - z_pump_start) / t_hit  # constant approach velocity

    def update(frame_idx):
        progress = frame_idx / float(total_frames)
        t = progress * duration_sec

        # 1. Pump position and synchronous delay parameter tau
        z_pump = z_pump_start + v_pump * t

        if t < t_hit:
            current_tau = delays[0] * (1.0 - t / t_hit)   # -1.5 ps -> 0.0 ps
        else:
            current_tau = delays[-1] * ((t - t_hit) / (duration_sec - t_hit)) # 0.0 ps -> 5.0 ps

        idx_tau = int(np.clip(np.searchsorted(delays, current_tau), 0, len(delays) - 1))

        # 2. Render Pump Pulses:
        # STRICT REQUIREMENT: COMPLETELY ABSORBED AT SAMPLE DISC (z = z_sample)
        # Pump NEVER emerges past z_sample!
        env_pump = np.exp(-0.5 * ((pts_pump - z_pump) / sigma_pump)**2)
        phase_pump = k_pump * (pts_pump - z_pump)
        # Mask strictly cuts off at sample and when pulse has entered
        mask_pump = (env_pump > 0.015) & (pts_pump < z_sample)

        # Sigma+ pump: counter-clockwise helix (absorbed at sample)
        Ex_pump_plus = np.where(mask_pump, env_pump * A_pump * np.cos(phase_pump), np.nan)
        Ey_pump_plus = np.where(mask_pump, env_pump * A_pump * np.sin(phase_pump), np.nan)
        line_pump_plus.set_data(Ex_pump_plus, Ey_pump_plus)
        line_pump_plus.set_3d_properties(pts_pump, zdir='x')

        # Sigma- pump: clockwise helix (absorbed at sample)
        Ex_pump_minus = np.where(mask_pump, env_pump * A_pump * np.cos(phase_pump), np.nan)
        Ey_pump_minus = np.where(mask_pump, -env_pump * A_pump * np.sin(phase_pump), np.nan)
        line_pump_minus.set_data(Ex_pump_minus, Ey_pump_minus)
        line_pump_minus.set_3d_properties(pts_pump, zdir='x')

        # 3. Render Stream of Smaller Probe Pulses:
        # Continuous pulse trains flowing along optical axis for both setups
        Ex_probe_plus  = np.full_like(pts_probe, np.nan)
        Ey_probe_plus  = np.full_like(pts_probe, np.nan)
        Ex_probe_minus = np.full_like(pts_probe, np.nan)
        Ey_probe_minus = np.full_like(pts_probe, np.nan)

        k_min = int(np.floor((v_probe * t - z_screen - 1.0) / d_probe))
        k_max = int(np.ceil((v_probe * t - (-0.5) + 1.0) / d_probe))

        tau_relax = 1.8

        for k in range(k_min, k_max + 1):
            z_pk = v_probe * t - k * d_probe
            if z_pk < -0.6 or z_pk > z_screen + 0.6:
                continue

            # Moment this packet crossed the sample disc (z = z_sample)
            t_cross_k = (z_sample + k * d_probe) / v_probe

            if t_cross_k < t_hit:
                # Crossed before pump: ground state transmission
                T_k_plus = 1.0
                T_k_minus = 1.0
            else:
                # Crossed after pump: sample demagnetized and relaxing!
                tau_k = delays[-1] * ((t_cross_k - t_hit) / (duration_sec - t_hit))
                tau_eff = max(0.0, tau_k)
                T_k_plus  = 1.0 - 0.92 * np.exp(-tau_eff / tau_relax)  # collapses to 8%!
                T_k_minus = 1.0 - 0.18 * np.exp(-tau_eff / tau_relax)  # stays large at 82%!

            # Local envelope for packet k
            mask_k = np.abs(pts_probe - z_pk) < 3.2 * sigma_probe
            if not np.any(mask_k):
                continue

            z_sub = pts_probe[mask_k]
            env_k = np.exp(-0.5 * ((z_sub - z_pk) / sigma_probe)**2)
            phase_k = k_probe * (z_sub - z_pk)

            valid = env_k > 0.015

            # Top Setup (Sigma+): heavy absorption collapse
            A_eff_plus = np.where(z_sub < z_sample, A_probe_in, A_probe_in * T_k_plus)
            Ex_sub_plus = np.where(valid, env_k * A_eff_plus * np.cos(phase_k), np.nan)
            Ey_sub_plus = np.where(valid, env_k * A_eff_plus * np.sin(phase_k), np.nan)
            Ex_probe_plus[mask_k] = np.where(valid, Ex_sub_plus, Ex_probe_plus[mask_k])
            Ey_probe_plus[mask_k] = np.where(valid, Ey_sub_plus, Ey_probe_plus[mask_k])

            # Bottom Setup (Sigma-): high probe transmission
            A_eff_minus = np.where(z_sub < z_sample, A_probe_in, A_probe_in * T_k_minus)
            Ex_sub_minus = np.where(valid, env_k * A_eff_minus * np.cos(phase_k), np.nan)
            Ey_sub_minus = np.where(valid, env_k * A_eff_minus * np.sin(phase_k), np.nan)
            Ex_probe_minus[mask_k] = np.where(valid, Ex_sub_minus, Ex_probe_minus[mask_k])
            Ey_probe_minus[mask_k] = np.where(valid, Ey_sub_minus, Ey_probe_minus[mask_k])

        line_probe_plus.set_data(Ex_probe_plus, Ey_probe_plus)
        line_probe_plus.set_3d_properties(pts_probe, zdir='x')

        line_probe_minus.set_data(Ex_probe_minus, Ey_probe_minus)
        line_probe_minus.set_3d_properties(pts_probe, zdir='x')

        # 4. Update 2D Graphs: Both traces evolve SIMULTANEOUSLY
        line_t_plus.set_data(delays[:idx_tau], T_plus_curve[:idx_tau])
        line_t_minus.set_data(delays[:idx_tau], T_minus_curve[:idx_tau])
        marker_t_plus.set_data([current_tau], [T_plus_curve[idx_tau]])
        marker_t_minus.set_data([current_tau], [T_minus_curve[idx_tau]])

        line_asym.set_data(delays[:idx_tau], Asym_curve[:idx_tau])
        marker_asym.set_data([current_tau], [Asym_curve[idx_tau]])

        vline_t.set_xdata([current_tau, current_tau])
        vline_asym.set_xdata([current_tau, current_tau])

        return (
            line_pump_plus, line_probe_plus, badge_plus,
            line_pump_minus, line_probe_minus, badge_minus,
            line_t_plus, line_t_minus, line_asym,
            marker_t_plus, marker_t_minus, marker_asym,
            vline_t, vline_asym
        )

    if preview_only:
        update(int(total_frames * 0.40))
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
    parser = argparse.ArgumentParser(description="XMCD Dual 3D Pump-Probe Simulation (Sigma+ vs Sigma-)")
    parser.add_argument("--preview", action="store_true", help="Save a single high-res preview frame")
    parser.add_argument("--output", type=str, default=None, help="Output filename (.mp4 or .gif)")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument(
        "--duration",
        type=float,
        default=16.0 / 1.2,
        help="Duration of simulation cycle in seconds (default ~13.3 s = 20%% faster than 16 s)",
    )
    args = parser.parse_args()

    build_simulation(
        output_path=args.output,
        preview_only=args.preview,
        fps=args.fps,
        duration_sec=args.duration
    )
