import os
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI


USER_FIXED_BETA = 0.9
USER_N_ALPHA    = 100


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


N_k = 2000


alpha_vec = np.linspace(0, 2, USER_N_ALPHA)


kx_vec = np.linspace(-np.pi, np.pi, N_k, endpoint=False)
ky_vec = np.linspace(0, 2 * np.pi, N_k, endpoint=False)
dkx = kx_vec[1] - kx_vec[0]
dky = ky_vec[1] - ky_vec[0]


e = 1.0
hbar = 1.0


SAVE_DIR = "figs_shg_1D_sumrule_optimized"
if rank == 0:
    os.makedirs(SAVE_DIR, exist_ok=True)


comm.Barrier()


def get_hamiltonian(kx, ky, alpha, beta):
    d1 = np.sin(kx) * np.cos(ky) + alpha * np.sin(kx) + beta * np.sin(2 * kx)
    d2 = np.sin(kx) * np.sin(ky)
    d3 = np.cos(kx)
    H = np.array([[d3, d1 - 1j * d2], [d1 + 1j * d2, -d3]], dtype=np.complex128)
    return H

def get_hamiltonian_derivatives(kx, ky, alpha, beta):

    dd1_dkx = np.cos(kx) * np.cos(ky) + alpha * np.cos(kx) + 2 * beta * np.cos(2 * kx)
    dd2_dkx = np.cos(kx) * np.sin(ky)
    dd3_dkx = -np.sin(kx)
    dH_dkx = np.array([[dd3_dkx, dd1_dkx - 1j * dd2_dkx], [dd1_dkx + 1j * dd2_dkx, -dd3_dkx]], dtype=np.complex128)

    dd1_dky = -np.sin(kx) * np.sin(ky)
    dd2_dky = np.sin(kx) * np.cos(ky)
    dd3_dky = 0
    dH_dky = np.array([[dd3_dky, dd1_dky - 1j * dd2_dky], [dd1_dky + 1j * dd2_dky, -dd3_dky]], dtype=np.complex128)
    return dH_dkx, dH_dky

def get_hamiltonian_second_derivatives(kx, ky, alpha, beta):

    d1_xx = -np.sin(kx) * np.cos(ky) - alpha * np.sin(kx) - 4 * beta * np.sin(2 * kx)
    d2_xx = -np.sin(kx) * np.sin(ky)
    d3_xx = -np.cos(kx)
    d2H_dkx2 = np.array([[d3_xx, d1_xx - 1j * d2_xx], [d1_xx + 1j * d2_xx, -d3_xx]], dtype=np.complex128)


    d1_yy = -np.sin(kx) * np.cos(ky)
    d2_yy = -np.sin(kx) * np.sin(ky)
    d3_yy = 0
    d2H_dky2 = np.array([[d3_yy, d1_yy - 1j * d2_yy], [d1_yy + 1j * d2_yy, -d3_yy]], dtype=np.complex128)


    d1_xy = -np.cos(kx) * np.sin(ky)
    d2_xy = np.cos(kx) * np.cos(ky)
    d3_xy = 0
    d2H_dkx_dky = np.array([[d3_xy, d1_xy - 1j * d2_xy], [d1_xy + 1j * d2_xy, -d3_xy]], dtype=np.complex128)

    return d2H_dkx2, d2H_dky2, d2H_dkx_dky


def calculate_eigensystem_on_grid(alpha, beta):
    E_bands_data = np.zeros((N_k, N_k, 2))
    u_bands_data = np.zeros((N_k, N_k, 2, 2), dtype=np.complex128)
    for i, kx in enumerate(kx_vec):
        for j, ky in enumerate(ky_vec):
            H = get_hamiltonian(kx, ky, alpha, beta)
            evals, evecs = np.linalg.eigh(H)
            sort_indices = np.argsort(evals)
            E_bands_data[i, j, :] = evals[sort_indices]
            u_bands_data[i, j, :, :] = evecs[:, sort_indices].T
    return E_bands_data, u_bands_data

def calculate_all_matrix_elements(alpha, beta):
    E_bands, u_bands = calculate_eigensystem_on_grid(alpha, beta)
    num_bands = 2
    v_matrix = np.zeros((N_k, N_k, num_bands, num_bands, 2), dtype=np.complex128)
    r_matrix = np.zeros((N_k, N_k, num_bands, num_bands, 2), dtype=np.complex128)

    w_matrix = np.zeros((N_k, N_k, num_bands, num_bands, 2, 2), dtype=np.complex128)

    for i, kx in enumerate(kx_vec):
        for j, ky in enumerate(ky_vec):

            dH_dkx, dH_dky = get_hamiltonian_derivatives(kx, ky, alpha, beta)

            d2H_xx, d2H_yy, d2H_xy = get_hamiltonian_second_derivatives(kx, ky, alpha, beta)
            d2H_yx = d2H_xy

            for n in range(num_bands):
                for m in range(num_bands):

                    v_matrix[i, j, n, m, 0] = np.vdot(u_bands[i, j, n, :], (dH_dkx / hbar) @ u_bands[i, j, m, :])
                    v_matrix[i, j, n, m, 1] = np.vdot(u_bands[i, j, n, :], (dH_dky / hbar) @ u_bands[i, j, m, :])


                    if n != m:
                        omega_nm = (E_bands[i, j, n] - E_bands[i, j, m]) / hbar
                        r_matrix[i, j, n, m, 0] = v_matrix[i, j, n, m, 0] / (1j * omega_nm)
                        r_matrix[i, j, n, m, 1] = v_matrix[i, j, n, m, 1] / (1j * omega_nm)


                    w_matrix[i, j, n, m, 0, 0] = np.vdot(u_bands[i, j, n, :], d2H_xx @ u_bands[i, j, m, :])
                    w_matrix[i, j, n, m, 1, 1] = np.vdot(u_bands[i, j, n, :], d2H_yy @ u_bands[i, j, m, :])
                    w_matrix[i, j, n, m, 0, 1] = np.vdot(u_bands[i, j, n, :], d2H_xy @ u_bands[i, j, m, :])
                    w_matrix[i, j, n, m, 1, 0] = np.vdot(u_bands[i, j, n, :], d2H_yx @ u_bands[i, j, m, :])

    return E_bands, v_matrix, r_matrix, u_bands, w_matrix


def F_plus(omega_val, omega, gamma):
    gamma_safe = max(gamma, 1e-10)
    return 1 / (omega_val - omega - 1j * gamma_safe) + 1 / (omega_val + omega + 1j * gamma_safe)

def F_minus(omega_val, omega, gamma):
    gamma_safe = max(gamma, 1e-10)
    return 1 / (omega_val - omega - 1j * gamma_safe) - 1 / (omega_val + omega + 1j * gamma_safe)

def calculate_chi_interband(E_bands, r_matrix, omega, gamma):

    components = ['xxx', 'yyy', 'xxy', 'xyy', 'yxx', 'yyx']
    cart_map = {'x': 0, 'y': 1}
    num_bands = 2
    f_n = np.array([1, 0])
    integrand = {comp: np.zeros((N_k, N_k), dtype=np.complex128) for comp in components}

    for i in range(N_k):
        for j in range(N_k):
            for n in range(num_bands):
                for m in range(num_bands):
                    f_nm = f_n[n] - f_n[m]
                    if abs(f_nm) < 1e-6: continue
                    omega_mn = (E_bands[i, j, m] - E_bands[i, j, n]) / hbar
                    for p in range(num_bands):
                        omega_mp = (E_bands[i, j, m] - E_bands[i, j, p]) / hbar
                        omega_pn = (E_bands[i, j, p] - E_bands[i, j, n]) / hbar
                        f_pm = f_n[p] - f_n[m]
                        f_np = f_n[n] - f_n[p]
                        L_mnp = (0.5 * f_pm * F_plus(omega_mp, omega, gamma) +
                                 0.5 * f_np * F_plus(omega_pn, omega, gamma) -
                                 f_nm * F_plus(omega_mn, 2 * omega, gamma))
                        denominator = omega_mp - 0.5 * omega_mn
                        term_base_ter = L_mnp / denominator
                        for comp in components:
                            a, b, c = [cart_map[s] for s in comp]
                            term1 = r_matrix[i, j, n, m, a] * r_matrix[i, j, m, p, b] * r_matrix[i, j, p, n, c] * term_base_ter
                            term2 = r_matrix[i, j, n, m, a] * r_matrix[i, j, m, p, c] * r_matrix[i, j, p, n, b] * term_base_ter
                            integrand[comp][i, j] += (e ** 3 / (4 * hbar ** 2)) * (term1 + term2)
    return integrand

def calculate_chi_intraband_appendix_b(E_bands, v_matrix, r_matrix, u_bands, w_matrix, omega, gamma):
    components = ['xxx', 'yyy', 'xxy', 'xyy', 'yxx', 'yyx']

    comp_map = {
        'xxx': (0, 0, 0), 'yyy': (1, 1, 1),
        'xxy': (0, 0, 1), 'xyy': (0, 1, 1),
        'yxx': (1, 0, 0), 'yyx': (1, 1, 0)
    }

    num_bands = 2
    f_n = np.array([1, 0])
    integrand = {comp: np.zeros((N_k, N_k), dtype=np.complex128) for comp in components}


    gen_derivs = np.zeros((2, 2), dtype=np.complex128)
    gen_derivs_swap = np.zeros((2, 2), dtype=np.complex128)
    Delta_nm = np.zeros(2, dtype=np.complex128)
    Delta_mn = np.zeros(2, dtype=np.complex128)

    for i in range(N_k):
        for j in range(N_k):
            for n in range(num_bands):
                for m in range(num_bands):
                    f_nm = f_n[n] - f_n[m]
                    if abs(f_nm) < 1e-6: continue

                    omega_nm = (E_bands[i, j, n] - E_bands[i, j, m]) / hbar
                    omega_mn = -omega_nm


                    inv_omega_nm = 1j / omega_nm
                    inv_omega_mn = 1j / omega_mn

                    for axis in range(2):
                        Delta_nm[axis] = v_matrix[i, j, n, n, axis] - v_matrix[i, j, m, m, axis]
                        Delta_mn[axis] = -Delta_nm[axis]

                    r_nm_vec = r_matrix[i, j, n, m, :]
                    r_mn_vec = r_matrix[i, j, m, n, :]


                    for a in range(2):
                        for b in range(2):

                            w_val = w_matrix[i, j, n, m, a, b]
                            num = (1j * r_nm_vec[a] * Delta_nm[b]) + \
                                  (1j * r_nm_vec[b] * Delta_nm[a]) - w_val
                            gen_derivs[a, b] = inv_omega_nm * num


                            w_val_swap = np.conj(w_matrix[i, j, n, m, a, b])
                            num_swap = (1j * r_mn_vec[a] * Delta_mn[b]) + \
                                       (1j * r_mn_vec[b] * Delta_mn[a]) - w_val_swap
                            gen_derivs_swap[a, b] = inv_omega_mn * num_swap


                    rho_beta, rho_gamma = 0.5, 0.5
                    Fp_om   = F_plus(omega_mn, omega, gamma)
                    Fp_2om  = F_plus(omega_mn, 2 * omega, gamma)
                    Fm_om   = F_minus(omega_mn, omega, gamma)

                    factor1 = (Fp_2om + rho_beta * Fp_om) / (rho_gamma * omega_mn)
                    factor2 = (Fp_2om + rho_gamma * Fp_om) / (rho_beta * omega_mn)
                    factor_II = ((rho_beta**2 * Fp_om - Fp_2om) / (rho_gamma**2 * omega_mn**2))
                    factor_III_1 = (Fp_2om - rho_beta * Fp_om) / (rho_gamma * omega_mn)
                    factor_III_2 = (Fp_2om - rho_gamma * Fp_om) / (rho_beta * omega_mn)
                    factor_III_3 = (Fm_om + Fm_om) / (2 * omega)


                    for comp in components:
                        a, b, c = comp_map[comp]


                        deriv_symm_val1 = gen_derivs[a, c] * r_mn_vec[b] + r_nm_vec[a] * gen_derivs_swap[b, c]
                        term_I_val = deriv_symm_val1 * factor1


                        deriv_symm_val2 = gen_derivs[a, b] * r_mn_vec[c] + r_nm_vec[a] * gen_derivs_swap[c, b]
                        term_I_val += deriv_symm_val2 * factor2


                        term_II_val = (r_nm_vec[a] * r_mn_vec[b] * Delta_mn[c] + \
                                       r_nm_vec[a] * r_mn_vec[c] * Delta_mn[b]) * factor_II


                        asym1 = r_nm_vec[a] * gen_derivs_swap[b, c] - gen_derivs[a, c] * r_mn_vec[b]
                        asym2 = r_nm_vec[a] * gen_derivs_swap[c, b] - gen_derivs[a, b] * r_mn_vec[c]
                        asym3 = r_nm_vec[c] * gen_derivs_swap[b, a] - gen_derivs[c, a] * r_mn_vec[b]

                        term_III_val = asym1 * factor_III_1 + \
                                       asym2 * factor_III_2 + \
                                       asym3 * factor_III_3

                        total_val = (1j * e**3 / (8 * hbar**2)) * f_nm * (term_I_val + 2 * term_II_val + term_III_val)
                        integrand[comp][i, j] += total_val

    return integrand

def calculate_shg_susceptibility_total(E_bands, v_matrix, r_matrix, u_bands, w_matrix, omega, gamma):
    integrand_inter = calculate_chi_interband(E_bands, r_matrix, omega, gamma)

    integrand_intra = calculate_chi_intraband_appendix_b(E_bands, v_matrix, r_matrix, u_bands, w_matrix, omega, gamma)
    integrand_total = {comp: integrand_inter[comp] + integrand_intra[comp] for comp in integrand_inter}
    chi_2_total = {comp: np.sum(integrand_total[comp]) * (dkx * dky) / ((2 * np.pi) ** 2) for comp in integrand_total}
    components = ['xxx', 'yyy', 'xxy', 'xyy', 'yxx', 'yyx']
    return [chi_2_total[comp] for comp in components]


if __name__ == '__main__':

    components = ['xxx', 'yyy', 'xxy', 'xyy', 'yxx', 'yyx']


    AU_TO_PM_V = 1


    omega_shg = 2
    gamma_shg = 0.01


    full_tasks = [(i, (alpha, USER_FIXED_BETA)) for i, alpha in enumerate(alpha_vec)]


    my_tasks = full_tasks[rank::size]


    local_results = []


    task_iterator = my_tasks
    for idx, (alpha, beta) in task_iterator:

        E_bands, v_matrix, r_matrix, u_bands, w_matrix = calculate_all_matrix_elements(alpha, beta)

        chi_values = calculate_shg_susceptibility_total(E_bands, v_matrix, r_matrix, u_bands, w_matrix, omega_shg, gamma_shg)
        local_results.append((idx, chi_values))


    all_results_list = comm.gather(local_results, root=0)

    if rank == 0:


        flat_results = []
        for res_list in all_results_list:
            flat_results.extend(res_list)


        flat_results.sort(key=lambda x: x[0])


        final_data = [item[1] for item in flat_results]


        shg_line_au = np.array(final_data)
        shg_line_pm_v = shg_line_au * AU_TO_PM_V


        out_abs  = os.path.join(SAVE_DIR, "abs")
        out_real = os.path.join(SAVE_DIR, "real")
        out_imag = os.path.join(SAVE_DIR, "imag")
        for d in [out_abs, out_real, out_imag]:
            os.makedirs(d, exist_ok=True)

        tag = f"beta{USER_FIXED_BETA:g}_omega{omega_shg:g}_sumrule"


        FONT_TITLE = 17
        FONT_LABEL = 19
        FONT_TICK_AXIS = 20


        for idx, comp in enumerate(components):
            data_comp = shg_line_pm_v[:, idx]

            def save_line_plot(x_data, y_data, ylabel_tex, filename, folder):
                fig = plt.figure(figsize=(7, 5))
                ax = fig.add_subplot(111)
                ax.plot(x_data, y_data, 'o-', linewidth=2, markersize=4)
                ax.set_title(ylabel_tex, fontsize=FONT_TITLE)
                ax.set_xlabel('α', fontsize=FONT_LABEL)
                ax.set_ylabel('SHG Intensity (arb. units)', fontsize=FONT_LABEL)
                ax.tick_params(axis='both', which='major', labelsize=FONT_TICK_AXIS)
                ax.grid(True, linestyle='--', alpha=0.6)
                ax.text(0.05, 0.95, f'$\\beta = {USER_FIXED_BETA}$', transform=ax.transAxes,
                        fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                fig.tight_layout()
                fpath = os.path.join(folder, filename)
                fig.savefig(fpath, dpi=300)
                plt.close(fig)

            y_abs = np.abs(data_comp)
            save_line_plot(alpha_vec, y_abs, f'$|\\chi_{{{comp}}}^{{(2)}}|$', f"{comp}_abs_{tag}.png", out_abs)

            y_real_abs = np.abs(np.real(data_comp))
            save_line_plot(alpha_vec, y_real_abs, f'$|\\mathrm{{Re}}[\\chi_{{{comp}}}^{{(2)}}]|$', f"{comp}_real_{tag}.png", out_real)

            y_imag_abs = np.abs(np.imag(data_comp))
            save_line_plot(alpha_vec, y_imag_abs, f'$|\\mathrm{{Im}}[\\chi_{{{comp}}}^{{(2)}}]|$', f"{comp}_imag_{tag}.png", out_imag)


        try:
            yxx_idx = components.index('yxx')
        except ValueError:
            exit(1)

        yxx_data = shg_line_pm_v[:, yxx_idx]
        yxx_real = np.real(yxx_data)
        yxx_real_abs = np.abs(yxx_real)
        yxx_imag = np.imag(yxx_data)
        yxx_imag_abs = np.abs(yxx_imag)
        yxx_total_abs = np.abs(yxx_data)

        data_to_save = np.column_stack([
            alpha_vec, yxx_real, yxx_real_abs, yxx_imag, yxx_imag_abs, yxx_total_abs
        ])

        header = "alpha,yxx_Real,abs_yxx_Real,yxx_Imag,abs_yxx_Imag,abs_yxx_Total"
        csv_filename = f"Method_SumRule_yxx_data_beta{USER_FIXED_BETA}_omega{omega_shg}.csv"
        csv_path = os.path.join(SAVE_DIR, csv_filename)
        np.savetxt(csv_path, data_to_save, delimiter=",", header=header, comments="")
