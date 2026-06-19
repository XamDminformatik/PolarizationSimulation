import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

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

def oke_sample_matrix(gamma_kerr, theta_pump):
    """
    Jones matrix for the sample exhibiting Optical Kerr Effect.
    Induced retardation gamma_kerr along the pump polarization direction theta_pump.
    """
    return retarder_matrix(gamma_kerr, theta_pump)

def simulate_propagation(hwp_angle, qwp_angle, pump_angle, kerr_retardation):
    """
    Calculate the Jones vector of the probe pulse at different stages.
    """
    # 1. Initial state (s-polarized, usually y-axis)
    E0 = np.array([0, 1], dtype=complex)

    # 2. After HWP
    J_hwp = hwp_matrix(hwp_angle)
    E1 = J_hwp @ E0

    # 3. After QWP
    J_qwp = qwp_matrix(qwp_angle)
    E2 = J_qwp @ E1

    # 4. After Sample (OKE)
    J_oke = oke_sample_matrix(kerr_retardation, pump_angle)
    E3 = J_oke @ E2

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

def main():
    fig = plt.figure(figsize=(10, 8))
    # Use orthogonal projection to prevent transverse waves from looking longitudinal due to perspective
    ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
    ax.view_init(elev=20, azim=-45)
    plt.subplots_adjust(bottom=0.35)

    # Setup parameters
    z_hwp = 2.0
    z_qwp = 4.0
    z_sample = 6.0
    z_end = 8.0

    # Initial state
    hwp_angle_init = np.pi / 8  # 22.5 deg
    qwp_angle_init = 0.0
    pump_angle_init = np.pi / 4 # 45 deg
    kerr_init = 0.5

    # Lines for wave segments
    line_0, = ax.plot([], [], [], color='blue', label='Initial (s-pol)')
    line_1, = ax.plot([], [], [], color='green', label='After HWP')
    line_2, = ax.plot([], [], [], color='red', label='After QWP')
    line_3, = ax.plot([], [], [], color='purple', label='After Sample (OKE)')

    ax.set_xlim(0, z_end)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)
    ax.set_xlabel('Propagation (z)')
    ax.set_ylabel('Ex')
    ax.set_zlabel('Ey')
    ax.legend()

    # Draw optical elements
    def draw_plate(z, name, color):
        y = np.linspace(-1.5, 1.5, 2)
        x = np.linspace(-1.5, 1.5, 2)
        x_grid, y_grid = np.meshgrid(x, y)
        z_grid = np.full_like(x_grid, z)
        ax.plot_surface(z_grid, x_grid, y_grid, alpha=0.3, color=color)
        ax.text(z, 0, 1.6, name, ha='center')

    draw_plate(z_hwp, 'HWP', 'cyan')
    draw_plate(z_qwp, 'QWP', 'magenta')
    draw_plate(z_sample, 'Sample', 'yellow')

    # Sliders
    axcolor = 'lightgoldenrodyellow'
    ax_hwp = plt.axes([0.15, 0.25, 0.65, 0.03], facecolor=axcolor)
    ax_qwp = plt.axes([0.15, 0.2, 0.65, 0.03], facecolor=axcolor)
    ax_pump = plt.axes([0.15, 0.15, 0.65, 0.03], facecolor=axcolor)
    ax_kerr = plt.axes([0.15, 0.1, 0.65, 0.03], facecolor=axcolor)

    s_hwp = Slider(ax_hwp, 'HWP Angle (rad)', 0.0, np.pi, valinit=hwp_angle_init)
    s_qwp = Slider(ax_qwp, 'QWP Angle (rad)', 0.0, np.pi, valinit=qwp_angle_init)
    s_pump = Slider(ax_pump, 'Pump Angle (rad)', 0.0, np.pi, valinit=pump_angle_init)
    s_kerr = Slider(ax_kerr, 'Kerr Retardation', 0.0, 2.0, valinit=kerr_init)

    def update(frame):
        t = frame / 20.0
        hwp_angle = s_hwp.val
        qwp_angle = s_qwp.val
        pump_angle = s_pump.val
        kerr_retardation = s_kerr.val

        E0, E1, E2, E3 = simulate_propagation(hwp_angle, qwp_angle, pump_angle, kerr_retardation)

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

        return line_0, line_1, line_2, line_3

    ani = FuncAnimation(fig, update, frames=200, interval=50, blit=False)
    plt.show()

if __name__ == "__main__":
    main()
