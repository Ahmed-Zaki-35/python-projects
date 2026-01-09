import numpy as np

# =====================================================
# BUS DATA
# bus_type: 1 = Slack, 2 = PV, 3 = PQ
# bus, type, P_spec, Q_spec, V_init, angle_init(deg)
# =====================================================
bus_data = np.array([
    [1, 1,  0.0,  0.0, 1.05,  0.0],   # Slack
    [2, 3,  -0.7928, -0.4856, 1.00,  0.0],   # PV
    [3, 3, -0.6905, -0.4093, 1.00,  0.0]    # PQ
])

nbus = len(bus_data)

# =====================================================
# Y-BUS MATRIX
# =====================================================
Ybus = np.array([
    [4-12j, -2+6j, -2+6j],
    [-2+6j, 3.25-9.75j, -1.25+3.75j],
    [-2+6j, -1.25+3.75j, 3.25-9.75j]
], dtype=complex)

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
tol = 1e-6
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

    max_mismatch = np.max(np.abs(mismatch))

    if max_mismatch < tol:
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

    # J1
    for i, m in enumerate(pvpq):
        for j, n in enumerate(pvpq):
            if m == n:
                J1[i,j] = -Q_calc[m]
            else:
                J1[i,j] = V[m]*V[n]*(G[m,n]*np.sin(delta[m]-delta[n]) -
                                     B[m,n]*np.cos(delta[m]-delta[n]))

    # J2
    for i, m in enumerate(pvpq):
        for j, n in enumerate(pq):
            if m == n:
                J2[i,j] = P_calc[m]/V[m]
            else:
                J2[i,j] = V[m]*(G[m,n]*np.cos(delta[m]-delta[n]) +
                                 B[m,n]*np.sin(delta[m]-delta[n]))

    # J3
    for i, m in enumerate(pq):
        for j, n in enumerate(pvpq):
            if m == n:
                J3[i,j] = P_calc[m]
            else:
                J3[i,j] = -V[m]*V[n]*(G[m,n]*np.cos(delta[m]-delta[n]) +
                                       B[m,n]*np.sin(delta[m]-delta[n]))

    # J4
    for i, m in enumerate(pq):
        for j, n in enumerate(pq):
            if m == n:
                J4[i,j] = Q_calc[m]/V[m]
            else:
                J4[i,j] = V[m]*(G[m,n]*np.sin(delta[m]-delta[n]) -
                                 B[m,n]*np.cos(delta[m]-delta[n]))

    J = np.block([[J1, J2],
                  [J3, J4]])

    try:
        dx = np.linalg.solve(J, mismatch)
    except np.linalg.LinAlgError:
        break

    delta[pvpq] += dx[:npv+npq]
    V[pq] += dx[npv+npq:]

# =====================================================
# RESULTS & STABILITY ASSESSMENT
# =====================================================
print("\n================ POWER FLOW RESULTS ================\n")

for i in range(nbus):
    print(f"Bus {i+1:>2} | V = {V[i]:.4f} pu | Angle = {np.rad2deg(delta[i]):.4f} deg")

print("\n================ STABILITY ANALYSIS =================\n")

if converged:
    print(f"✔ Load Flow Converged in {iterations} iterations")
    print(f"✔ Maximum mismatch = {max_mismatch:.6e}")

    if np.all((V >= 0.9) & (V <= 1.1)):
        print("✔ Voltage profile within acceptable limits (0.9–1.1 pu)")
        print("\nSYSTEM STATUS: STABLE")
    else:
        print("✖ Voltage limit violation detected")
        print("\nSYSTEM STATUS: MARGINALLY UNSTABLE")

else:
    print("✖ Load Flow DID NOT converge")
    print("\nSYSTEM STATUS: UNSTABLE")

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