import numpy as np

# ===============================
# Bus Data
# ===============================
# bus_type: 1 = Slack, 2 = PV, 3 = PQ
# bus, type, P_spec, Q_spec, V_mag, V_ang(deg)
bus_data = np.array([
    [1, 1,  0.0,  0.0, 1.06,  0.0],   # Slack
    [2, 2,  0.5,  0.0, 1.04,  0.0],   # PV  generator injection bus
    [3, 3, -0.6, -0.3, 1.00,  0.0]    # PQ  Load injection bus
])

nbus = len(bus_data)

# ===============================
# Y-Bus Matrix (example)
# ===============================
Ybus = np.array([
    [10-30j, -5+15j, -5+15j],
    [-5+15j, 10-30j, -5+15j],
    [-5+15j, -5+15j, 10-30j]
], dtype=complex)

G = np.real(Ybus)
B = np.imag(Ybus)

# ===============================
# Initialization
# ===============================
V = bus_data[:,4].astype(float)
delta = np.deg2rad(bus_data[:,5].astype(float))

P_spec = bus_data[:,2]
Q_spec = bus_data[:,3]
bus_type = bus_data[:,1]

slack = np.where(bus_type == 1)[0]
pv = np.where(bus_type == 2)[0]
pq = np.where(bus_type == 3)[0]

max_iter = 20
tol = 1e-6

# ===============================
# Power Calculation Function
# ===============================
def power_calc(V, delta):
    P = np.zeros(nbus)
    Q = np.zeros(nbus)

    for i in range(nbus):
        for k in range(nbus):
            P[i] += V[i]*V[k]*(G[i,k]*np.cos(delta[i]-delta[k]) +
                               B[i,k]*np.sin(delta[i]-delta[k]))
            Q[i] += V[i]*V[k]*(G[i,k]*np.sin(delta[i]-delta[k]) -
                               B[i,k]*np.cos(delta[i]-delta[k]))
    return P, Q

# ===============================
# Newton-Raphson Iteration
# ===============================
for iteration in range(max_iter):

    P_calc, Q_calc = power_calc(V, delta)

    # Mismatch
    dP = P_spec - P_calc
    dQ = Q_spec - Q_calc

    mismatch = np.concatenate((dP[pv.tolist() + pq.tolist()],
                                dQ[pq]))

    if np.max(np.abs(mismatch)) < tol:
        print(f"\nConverged in {iteration+1} iterations")
        break

    # ===============================
    # Jacobian Matrix
    # ===============================
    npv = len(pv)
    npq = len(pq)

    J1 = np.zeros((npv+npq, npv+npq))
    J2 = np.zeros((npv+npq, npq))
    J3 = np.zeros((npq, npv+npq))
    J4 = np.zeros((npq, npq))

    pvpq = np.concatenate((pv, pq))

    # J1: dP/dδ
    for i, m in enumerate(pvpq):
        for j, n in enumerate(pvpq):
            if m == n:
                for k in range(nbus):
                    J1[i,j] += V[m]*V[k]*(-G[m,k]*np.sin(delta[m]-delta[k])
                                           + B[m,k]*np.cos(delta[m]-delta[k]))
                J1[i,j] -= Q_calc[m]
            else:
                J1[i,j] = V[m]*V[n]*(G[m,n]*np.sin(delta[m]-delta[n])
                                      - B[m,n]*np.cos(delta[m]-delta[n]))

    # J2: dP/dV
    for i, m in enumerate(pvpq):
        for j, n in enumerate(pq):
            if m == n:
                for k in range(nbus):
                    J2[i,j] += V[k]*(G[m,k]*np.cos(delta[m]-delta[k]) +
                                     B[m,k]*np.sin(delta[m]-delta[k]))
                J2[i,j] += P_calc[m]/V[m]
            else:
                J2[i,j] = V[m]*(G[m,n]*np.cos(delta[m]-delta[n]) +
                                 B[m,n]*np.sin(delta[m]-delta[n]))

    # J3: dQ/dδ
    for i, m in enumerate(pq):
        for j, n in enumerate(pvpq):
            if m == n:
                for k in range(nbus):
                    J3[i,j] += V[m]*V[k]*(G[m,k]*np.cos(delta[m]-delta[k]) +
                                           B[m,k]*np.sin(delta[m]-delta[k]))
                J3[i,j] -= P_calc[m]
            else:
                J3[i,j] = -V[m]*V[n]*(G[m,n]*np.cos(delta[m]-delta[n]) +
                                       B[m,n]*np.sin(delta[m]-delta[n]))

    # J4: dQ/dV
    for i, m in enumerate(pq):
        for j, n in enumerate(pq):
            if m == n:
                for k in range(nbus):
                    J4[i,j] += V[k]*(G[m,k]*np.sin(delta[m]-delta[k]) -
                                     B[m,k]*np.cos(delta[m]-delta[k]))
                J4[i,j] -= Q_calc[m]/V[m]
            else:
                J4[i,j] = V[m]*(G[m,n]*np.sin(delta[m]-delta[n]) -
                                 B[m,n]*np.cos(delta[m]-delta[n]))

    # Full Jacobian
    J = np.block([[J1, J2],
                  [J3, J4]])

    # ===============================
    # Solve Corrections
    # ===============================
    dx = np.linalg.solve(J, mismatch)

    d_delta = dx[:npv+npq]
    d_V = dx[npv+npq:]

    delta[pvpq] += d_delta
    V[pq] += d_V

# ===============================
# Final Results
# ===============================
print("\nBus Results:")
for i in range(nbus):
    print(f"Bus {i+1}: |V| = {V[i]:.4f} pu, δ = {np.rad2deg(delta[i]):.4f} deg")

