import numpy as np

# =====================================================
# BUS DATA
# bus_type: 1 = Slack, 2 = PV, 3 = PQ
# bus, type, P_spec, Q_spec, V_init, angle_init(deg)
# =====================================================
bus_data = np.array([
    [1, 1,  0.0,  0.0, 1.06,  0.0],   # Slack
    [2, 2,  0.5,  0.0, 1.04,  0.0],   # PV
    [3, 3, -0.6, -0.3, 1.00,  0.0],   # PQ
    [4, 3, -0.4, -0.2, 1.00,  0.0]    # PQ
])

nbus = len(bus_data)

# =====================================================
# LINE DATA (MESH SYSTEM)
# from, to, R(pu), X(pu), B/2(pu)
# =====================================================
line_data = [
    [1, 2, 0.02, 0.06, 0.03],
    [1, 3, 0.08, 0.24, 0.025],
    [2, 3, 0.06, 0.18, 0.02],   # Loop
    [2, 4, 0.06, 0.18, 0.02],
    [3, 4, 0.01, 0.03, 0.015]  # Loop
]

# =====================================================
# Y-BUS CONSTRUCTION (RADIAL + MESH)
# =====================================================
def build_ybus(nbus, line_data):
    Ybus = np.zeros((nbus, nbus), dtype=complex)

    for line in line_data:
        i, j, R, X, B = line
        i -= 1
        j -= 1

        z = complex(R, X)
        y = 1 / z
        b_shunt = complex(0, B)

        Ybus[i, j] -= y
        Ybus[j, i] -= y

        Ybus[i, i] += y + b_shunt
        Ybus[j, j] += y + b_shunt

    return Ybus

Ybus = build_ybus(nbus, line_data)
G = np.real(Ybus)
B = np.imag(Ybus)

# =====================================================
# INITIALIZATION
# =====================================================
V = bus_data[:, 4].astype(float)
delta = np.deg2rad(bus_data[:, 5].astype(float))

P_spec = bus_data[:, 2]
Q_spec = bus_data[:, 3]
bus_type = bus_data[:, 1]

slack = np.where(bus_type == 1)[0]
pv = np.where(bus_type == 2)[0]
pq = np.where(bus_type == 3)[0]

max_iter = 20
tol = 1e-10
converged = False

# =====================================================
# POWER CALCULATION
# =====================================================
def calc_power(V, delta):
    P = np.zeros(nbus)
    Q = np.zeros(nbus)

    for i in range(nbus):
        for k in range(nbus):
            P[i] += V[i]*V[k]*(G[i,k]*np.cos(delta[i]-delta[k]) +
                               B[i,k]*np.sin(delta[i]-delta[k]))
            Q[i] += V[i]*V[k]*(G[i,k]*np.sin(delta[i]-delta[k]) -
                               B[i,k]*np.cos(delta[i]-delta[k]))
    return P, Q

# =====================================================
# NEWTON–RAPHSON ITERATION
# =====================================================
for it in range(max_iter):

    P_calc, Q_calc = calc_power(V, delta)

    dP = P_spec - P_calc
    dQ = Q_spec - Q_calc

    mismatch = np.concatenate((dP[np.r_[pv, pq]], dQ[pq]))

    if np.max(np.abs(mismatch)) < tol:
        converged = True
        iterations = it + 1
        break

    npv = len(pv)
    npq = len(pq)
    pvpq = np.r_[pv, pq]

    J1 = np.zeros((npv+npq, npv+npq))
    J2 = np.zeros((npv+npq, npq))
    J3 = np.zeros((npq, npv+npq))
    J4 = np.zeros((npq, npq))

    # J1: dP/dδ
    for i, m in enumerate(pvpq):
        for j, n in enumerate(pvpq):
            if m == n:
                J1[i,j] = -Q_calc[m]
            else:
                J1[i,j] = V[m]*V[n]*(G[m,n]*np.sin(delta[m]-delta[n]) -
                                     B[m,n]*np.cos(delta[m]-delta[n]))

    # J2: dP/dV
    for i, m in enumerate(pvpq):
        for j, n in enumerate(pq):
            if m == n:
                J2[i,j] = P_calc[m]/V[m]
            else:
                J2[i,j] = V[m]*(G[m,n]*np.cos(delta[m]-delta[n]) +
                                 B[m,n]*np.sin(delta[m]-delta[n]))

    # J3: dQ/dδ
    for i, m in enumerate(pq):
        for j, n in enumerate(pvpq):
            if m == n:
                J3[i,j] = P_calc[m]
            else:
                J3[i,j] = -V[m]*V[n]*(G[m,n]*np.cos(delta[m]-delta[n]) +
                                       B[m,n]*np.sin(delta[m]-delta[n]))

    # J4: dQ/dV
    for i, m in enumerate(pq):
        for j, n in enumerate(pq):
            if m == n:
                J4[i,j] = Q_calc[m]/V[m]
            else:
                J4[i,j] = V[m]*(G[m,n]*np.sin(delta[m]-delta[n]) -
                                 B[m,n]*np.cos(delta[m]-delta[n]))

    J = np.block([[J1, J2],
                  [J3, J4]])

    dx = np.linalg.solve(J, mismatch)

    delta[pvpq] += dx[:npv+npq]
    V[pq] += dx[npv+npq:]

# =====================================================
# RESULTS & STABILITY
# =====================================================
print("\n================ POWER FLOW RESULTS ================\n")

for i in range(nbus):
    print(f"Bus {i+1:>2} | V = {V[i]:.4f} pu | Angle = {np.rad2deg(delta[i]):.4f} deg")

print("\n================ STABILITY STATUS ==================\n")

if converged:
    print(f"✔ Converged in {iterations} iterations")
    if np.all((V >= 0.9) & (V <= 1.1)):
        print("✔ Voltage limits satisfied")
        print("SYSTEM STATUS: STABLE")
    else:
        print("✖ Voltage limit violation")
        print("SYSTEM STATUS: MARGINALLY UNSTABLE")
else:
    print("✖ Did not converge")
    print("SYSTEM STATUS: UNSTABLE")
# =====================================================
# POWER OUTPUT & MISMATCH RESULTS
# =====================================================
P_calc, Q_calc = calc_power(V, delta)

dP_final = P_spec - P_calc
dQ_final = Q_spec - Q_calc

print("\n================ POWER OUTPUT ======================\n")

print("Bus |   P (pu)    |   Q (pu)")
print("--------------------------------")
for i in range(nbus):
    print(f"{i+1:>3} | {P_calc[i]:>9.5f} | {Q_calc[i]:>9.5f}")

print("\n================ POWER MISMATCH ====================\n")

print("Bus | ΔP (pu)     | ΔQ (pu)")
print("--------------------------------")