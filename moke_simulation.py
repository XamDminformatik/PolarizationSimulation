import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

def rotation_matrix(theta):
    """Jones matrix for a circular retarder (optical rotation)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def retarder_matrix(gamma, theta):
    """
    Jones matrix for a linear retarder with retardation gamma
    and fast axis at angle theta.
    """
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    R_inv = np.array([[c, s], [-s, c]])

    # Phase matrix
    P = np.array([[np.exp(1j * gamma / 2), 0],
                  [0, np.exp(-1j * gamma / 2)]])

    return R @ P @ R_inv

def hwp_matrix(theta):
    """Jones matrix for a Half-Wave Plate."""
    return retarder_matrix(np.pi, theta)

def qwp_matrix(theta):
    """Jones matrix for a Quarter-Wave Plate."""
    return retarder_matrix(np.pi / 2, theta)

def sample_matrix(gamma_kerr, theta_pump, theta_ife):
    """
    Jones matrix for the sample exhibiting Optical Kerr Effect (OKE)
    and Inverse Faraday Effect (IFE).
    OKE is linear retardation gamma_kerr along theta_pump.
    IFE is circular retardation (rotation) theta_ife.
    """
    # Small effects commute approximately, so we can just multiply them
    M_oke = retarder_matrix(gamma_kerr, theta_pump)
    M_ife = rotation_matrix(theta_ife)
    return M_ife @ M_oke

def simulate_propagation(hwp_angle, qwp_angle, pump_angle, kerr_retardation, ife_rotation):
    """
    Calculate the Jones vector of the probe pulse at different stages.
    """
    E0 = np.array([0, 1], dtype=complex)

    J_hwp = hwp_matrix(hwp_angle)
    E1 = J_hwp @ E0

    J_qwp = qwp_matrix(qwp_angle)
    E2 = J_qwp @ E1

    J_sample = sample_matrix(kerr_retardation, pump_angle, ife_rotation)
    E3 = J_sample @ E2

    return E0, E1, E2, E3

def generate_wave(E, z_start, z_end, t, num_points=100, k=2*np.pi, omega=2*np.pi):
    """Generate 3D wave points for a given Jones vector."""
    z = np.linspace(z_start, z_end, num_points)
    # E = [Ex, Ey]
    # E(z, t) = Re{ E * exp(i(kz - omega*t)) }
    phase = k * z - omega * t
    Ex = np.real(E[0] * np.exp(1j * phase))
    Ey = np.real(E[1] * np.exp(1j * phase))
    return z, Ex, Ey

def calculate_polarization_parameters(E):
    """Calculate ellipticity angle (chi) and polarization angle (psi) from a Jones vector."""
    S1 = np.abs(E[0])**2 - np.abs(E[1])**2
    S2 = 2 * np.real(E[0] * np.conj(E[1]))
    S3 = 2 * np.imag(E[0] * np.conj(E[1]))

    psi = 0.5 * np.arctan2(S2, S1)
    chi = 0.5 * np.arcsin(S3) # assuming S0 ~ 1 since we don't do absorption
    return psi, chi

def ang_diff(a1, a2):
    """Helper to find the shortest angular difference."""
    diff = a1 - a2
    return (diff + np.pi/2) % np.pi - np.pi/2

def calculate_rotations_over_qwp(hwp_angle, pump_angle, gamma_kerr, theta_ife):
    """Calculate the input polarization and output rotations over a full range of QWP angles."""
    qwp_angles = np.linspace(0, np.pi, 100)

    chi_in_list = []
    psi_in_list = []

    rot_oke = []
    rot_ife = []
    rot_sum = []

    E0 = np.array([0, 1], dtype=complex)
    J_hwp = hwp_matrix(hwp_angle)
    E1 = J_hwp @ E0

    M_oke = retarder_matrix(gamma_kerr, pump_angle)
    M_ife = rotation_matrix(theta_ife)
    M_sum = M_ife @ M_oke

    for qwp in qwp_angles:
        J_qwp = qwp_matrix(qwp)
        E2 = J_qwp @ E1

        psi_in, chi_in = calculate_polarization_parameters(E2)
        chi_in_list.append(chi_in)
        psi_in_list.append(psi_in)

        # OKE only
        E_oke = M_oke @ E2
        psi_oke, _ = calculate_polarization_parameters(E_oke)

        # IFE only
        E_ife = M_ife @ E2
        psi_ife, _ = calculate_polarization_parameters(E_ife)

        # Total
        E_sum = M_sum @ E2
        psi_sum, _ = calculate_polarization_parameters(E_sum)

        rot_oke.append(ang_diff(psi_oke, psi_in))
        rot_ife.append(ang_diff(psi_ife, psi_in))
        rot_sum.append(ang_diff(psi_sum, psi_in))

    return qwp_angles, chi_in_list, rot_oke, rot_ife, rot_sum

def main():
    fig = plt.figure(figsize=(15, 10))

    # 3D Plot
    ax_3d = fig.add_subplot(2, 2, 1, projection='3d', proj_type='ortho')
    ax_3d.view_init(elev=20, azim=-45)

    # 2D Plot 1: Input to Sample (Ellipticity vs QWP angle)
    ax_in = fig.add_subplot(2, 2, 2)

    # 2D Plot 2: Output from Sample (Rotation vs QWP angle)
    ax_out = fig.add_subplot(2, 2, 4)

    plt.subplots_adjust(bottom=0.3, hspace=0.3, wspace=0.3)

    # --- 3D Plot Setup ---
    z_hwp, z_qwp, z_sample, z_end = 2.0, 4.0, 6.0, 8.0

    line_0, = ax_3d.plot([], [], [], color='blue', label='Initial')
    line_1, = ax_3d.plot([], [], [], color='green', label='After HWP')
    line_2, = ax_3d.plot([], [], [], color='red', label='After QWP')
    line_3, = ax_3d.plot([], [], [], color='purple', label='After Sample')

    ax_3d.set_xlim(0, z_end)
    ax_3d.set_ylim(-1.5, 1.5)
    ax_3d.set_zlim(-1.5, 1.5)
    ax_3d.set_title('3D Wave Propagation')

    def draw_plate(ax, z, name, color):
        y = np.linspace(-1.5, 1.5, 2)
        x = np.linspace(-1.5, 1.5, 2)
        x_grid, y_grid = np.meshgrid(x, y)
        z_grid = np.full_like(x_grid, z)
        ax.plot_surface(z_grid, x_grid, y_grid, alpha=0.3, color=color)
        ax.text(z, 0, 1.6, name, ha='center')

    draw_plate(ax_3d, z_hwp, 'HWP', 'cyan')
    draw_plate(ax_3d, z_qwp, 'QWP', 'magenta')
    draw_plate(ax_3d, z_sample, 'Sample', 'yellow')

    # --- 2D Plots Setup ---
    line_chi, = ax_in.plot([], [], label='Ellipticity entering sample')
    ax_in.set_title('Input to Sample')
    ax_in.set_xlabel('Current QWP Angle (rad)')
    ax_in.set_ylabel('Ellipticity angle (rad)')
    ax_in.set_xlim(0, np.pi)
    ax_in.set_ylim(-np.pi/4 - 0.1, np.pi/4 + 0.1)
    ax_in.legend()
    # Marker for current QWP position
    marker_in, = ax_in.plot([], [], 'ro')

    line_oke, = ax_out.plot([], [], label='Rotation (OKE)')
    line_ife, = ax_out.plot([], [], label='Rotation (IFE)')
    line_sum, = ax_out.plot([], [], label='Rotation (Total)', linestyle='--')
    ax_out.set_title('Output Rotation')
    ax_out.set_xlabel('Current QWP Angle (rad)')
    ax_out.set_ylabel('Rotation (rad)')
    ax_out.set_xlim(0, np.pi)
    ax_out.set_ylim(-0.5, 0.5) # Will be dynamically updated
    ax_out.legend()
    # Marker for current QWP position
    marker_out, = ax_out.plot([], [], 'ro')

    # --- Sliders ---
    axcolor = 'lightgoldenrodyellow'
    ax_hwp = plt.axes([0.15, 0.20, 0.65, 0.02], facecolor=axcolor)
    ax_qwp = plt.axes([0.15, 0.16, 0.65, 0.02], facecolor=axcolor)
    ax_pump = plt.axes([0.15, 0.12, 0.65, 0.02], facecolor=axcolor)
    ax_kerr = plt.axes([0.15, 0.08, 0.65, 0.02], facecolor=axcolor)
    ax_ife = plt.axes([0.15, 0.04, 0.65, 0.02], facecolor=axcolor)

    s_hwp = Slider(ax_hwp, 'HWP (rad)', 0.0, np.pi, valinit=np.pi/8)
    s_qwp = Slider(ax_qwp, 'QWP (rad)', 0.0, np.pi, valinit=0.0)
    s_pump = Slider(ax_pump, 'Pump (rad)', 0.0, np.pi, valinit=0.0)
    s_kerr = Slider(ax_kerr, 'Kerr Ret.', 0.0, 1.0, valinit=0.2)
    s_ife = Slider(ax_ife, 'IFE Rot.', -0.5, 0.5, valinit=0.1)

    def update(frame):
        t = frame / 20.0
        hwp_angle = s_hwp.val
        qwp_angle = s_qwp.val
        pump_angle = s_pump.val
        kerr_retardation = s_kerr.val
        ife_rotation = s_ife.val

        # Update 3D
        E0, E1, E2, E3 = simulate_propagation(hwp_angle, qwp_angle, pump_angle, kerr_retardation, ife_rotation)

        z0, Ex0, Ey0 = generate_wave(E0, 0, z_hwp, t)
        line_0.set_data(z0, Ex0)
        line_0.set_3d_properties(Ey0)

        z1, Ex1, Ey1 = generate_wave(E1, z_hwp, z_qwp, t)
        line_1.set_data(z1, Ex1)
        line_1.set_3d_properties(Ey1)

        z2, Ex2, Ey2 = generate_wave(E2, z_qwp, z_sample, t)
        line_2.set_data(z2, Ex2)
        line_2.set_3d_properties(Ey2)

        z3, Ex3, Ey3 = generate_wave(E3, z_sample, z_end, t)
        line_3.set_data(z3, Ex3)
        line_3.set_3d_properties(Ey3)

        # Update 2D lines only if parameters changed (optimization could go here, but we'll do it every frame for simplicity)
        qwp_range, chi_in, rot_oke, rot_ife, rot_sum = calculate_rotations_over_qwp(hwp_angle, pump_angle, kerr_retardation, ife_rotation)

        line_chi.set_data(qwp_range, chi_in)
        line_oke.set_data(qwp_range, rot_oke)
        line_ife.set_data(qwp_range, rot_ife)
        line_sum.set_data(qwp_range, rot_sum)

        # Dynamically adjust y-limits for the output rotation plot
        max_rot = max(np.max(np.abs(rot_sum)), np.max(np.abs(rot_oke)), np.max(np.abs(rot_ife)))
        max_rot = max(max_rot * 1.2, 0.05) # Add 20% margin, with a minimum of 0.05
        ax_out.set_ylim(-max_rot, max_rot)

        # Update markers for current QWP angle
        # Find index closest to current qwp_angle
        idx = np.argmin(np.abs(qwp_range - qwp_angle))
        marker_in.set_data([qwp_range[idx]], [chi_in[idx]])
        marker_out.set_data([qwp_range[idx]], [rot_sum[idx]])

        return line_0, line_1, line_2, line_3, line_chi, line_oke, line_ife, line_sum, marker_in, marker_out

    ani = FuncAnimation(fig, update, frames=200, interval=50, blit=False)
    plt.show()

if __name__ == "__main__":
    main()
