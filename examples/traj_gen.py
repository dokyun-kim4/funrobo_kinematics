
import numpy as np

from funrobo_kinematics.core.trajectory_generator import MultiAxisTrajectoryGenerator, MultiSegmentTrajectoryGenerator




class CubicPolynomial():
    """
    Cubic interpolation with position and velocity boundary constraints.
    """

    def __init__(self, ndof=None):
        """
        Initialize the trajectory generator.
        """
        self.ndof = ndof

    
    def solve(self, q0, qf, qd0, qdf, qdd0, qddf, T):
        """
        Compute cubic polynomial coefficients for each DOF.

        Parameters
        ----------
        q0 : array-like, shape (ndof,)
            Initial positions.
        qf : array-like, shape (ndof,)
            Final positions.
        qd0 : array-like or None, shape (ndof,)
            Initial velocities. If None, assumed zero.
        qdf : array-like or None, shape (ndof,)
            Final velocities. If None, assumed zero.
        T : float
            Total trajectory duration.
        """
        t0, tf = 0, T
        q0 = np.asarray(q0, dtype=float)
        qf = np.asarray(qf, dtype=float)
        qd0 = np.zeros_like(q0) if qd0 is None else np.asarray(qd0, dtype=float)
        qdf = np.zeros_like(q0) if qdf is None else np.asarray(qdf, dtype=float)
        
        A = np.array(
                [[1, t0, t0**2, t0**3],
                 [0, 1, 2*t0, 3*t0**2],
                 [1, tf, tf**2, tf**3],
                 [0, 1, 2*tf, 3*tf**2]
                ])

        b = np.vstack([
            q0,
            qd0,
            qf,
            qdf
        ])

        print(A.shape, b.shape)
        self.coeff = np.linalg.solve(A, b)
        

    def generate(self, t0=0, tf=0, nsteps=100):
        """
        Generate position, velocity, and acceleration trajectories.

        Parameters
        ----------
        t0 : float
            Start time.
        tf : float
            End time.
        nsteps : int
            Number of time samples.
        """
        t = np.linspace(t0, tf, nsteps)
        X = np.zeros((self.ndof, 3, len(t))) # type: ignore
        for i in range(self.ndof): # iterate through all DOFs # type: ignore
            c = self.coeff[:, i]

            q = c[0] + c[1] * t + c[2] * t**2 + c[3] * t**3
            qd = c[1] + 2 * c[2] * t + 3 * c[3] * t**2
            qdd = 2 * c[2] + 6 * c[3] * t

            X[i, 0, :] = q      # position
            X[i, 1, :] = qd     # velocity
            X[i, 2, :] = qdd    # acceleration

        return t, X


class QuinticPolynomial():
    """
    Quintic interpolation with position, velocity, and acceleration boundary constraints.
    """

    def __init__(self, ndof=None):
        """
        Initialize the trajectory generator.
        """
        self.ndof = ndof

    
    def solve(self, q0, qf, qd0, qdf, qdd0, qddf, T):
        """
        Compute quintic polynomial coefficients for each DOF.

        Parameters
        ----------
        q0 : array-like, shape (ndof,)
            Initial positions.
        qf : array-like, shape (ndof,)
            Final positions.
        qd0 : array-like or None, shape (ndof,)
            Initial velocities. If None, assumed zero.
        qdf : array-like or None, shape (ndof,)
            Final velocities. If None, assumed zero.
        qdd0 : array-like or None, shape (ndof,)
            Initial accelerations. If None, assumed zero.
        qddf : array-like or None, shape (ndof,)
            Final accelerations. If None, assumed zero.
        T : float
            Total trajectory duration.
        """
        t0, tf = 0, T
        q0 = np.asarray(q0, dtype=float)
        qf = np.asarray(qf, dtype=float)
        qd0 = np.zeros_like(q0) if qd0 is None else np.asarray(qd0, dtype=float)
        qdf = np.zeros_like(q0) if qdf is None else np.asarray(qdf, dtype=float)
        qdd0 = np.zeros_like(q0) if qdd0 is None else np.asarray(qdd0, dtype=float)
        qddf = np.zeros_like(q0) if qddf is None else np.asarray(qddf, dtype=float)

        A = np.array(
            [
            [1, t0, t0**2, t0**3, t0**4, t0**5],
            [0, 1, 2*t0, 3*t0**2, 4*t0**3, 5*t0**4],
            [0, 0, 2, 6*t0, 12*t0**2, 20*t0**3],
            [1, tf, tf**2, tf**3, tf**4, tf**5],
            [0, 1, 2*tf, 3*tf**2, 4*tf**3, 5*tf**4],
            [0, 0, 2, 6*tf, 12*tf**2, 20*tf**3]         
            ]
        )

        b = np.vstack([
            q0,
            qd0,
            qdd0,
            qf,
            qdf,
            qddf
        ])
        self.coeff = np.linalg.solve(A, b)
    
    def generate(self, t0=0, tf=0, nsteps=100):
        """
        Generate position, velocity, and acceleration trajectories.

        Parameters
        ----------
        t0 : float
            Start time.
        tf : float
            End time.
        nsteps : int
            Number of time samples.
        """
        t = np.linspace(t0, tf, nsteps)
        X = np.zeros((self.ndof, 3, len(t))) # type: ignore
        for i in range(self.ndof): # iterate through all DOFs # type: ignore
            c = self.coeff[:, i]

            q = c[0] + c[1] * t + c[2] * t**2 + c[3] * t**3 + c[4] * t**4 + c[5] * t**5
            qd = c[1] + 2 * c[2] * t + 3 * c[3] * t**2 + 4 * c[4] * t**3 + 5 * c[5] * t**4
            qdd = 2 * c[2] + 6 * c[3] * t + 12 * c[4] * t**2 + 20 * c[5] * t**3

            print(q.shape, type(q))
            X[i, 0, :] = q      # position
            X[i, 1, :] = qd     # velocity
            X[i, 2, :] = qdd    # acceleration

        return t, X

class Trapezoidal():
    def __init__(self, ndof=None):
        """
        Initialize the trajectory generator.
        """
        self.ndof = ndof

    def solve(self, q0, qf, qd0, qdf, qdd0, qddf, T):
        
        t0, tf = 0, T
        q0 = np.asarray(q0, dtype=float)
        qf = np.asarray(qf, dtype=float)
        qd0 = np.zeros_like(q0) if qd0 is None else np.asarray(qd0, dtype=float)
        qdf = np.zeros_like(q0) if qdf is None else np.asarray(qdf, dtype=float)
        qdd0 = np.zeros_like(q0) if qdd0 is None else np.asarray(qdd0, dtype=float)
        qddf = np.zeros_like(q0) if qddf is None else np.asarray(qddf, dtype=float)


        V_max = 1.5 * (qf - q0) / tf
        t_b = (q0 - qf + V_max * tf) / V_max
        accl = V_max / t_b

        self.q0, self.qf = q0, qf
        self.coeff = np.vstack([[q0, qf, V_max, t_b, accl]])

    def generate(self, t0=0, tf=0, nsteps=100):
        t = np.linspace(t0, tf, nsteps)
        X = np.zeros((self.ndof, 3, len(t))) # type: ignore
        for i in range(self.ndof): # iterate through all DOFs # type: ignore
            q0, qf, V_max, t_b, accl = self.coeff[:, i]
            q_arr = np.array([])
            qd_arr = np.array([])
            qdd_arr = np.array([])
            for t_ in t:
                print(t_, t_b)
                if 0 <= t_ < t_b:
                    print("acceleration phase")
                    q = q0 + accl*t_**2 / 2
                    qd = accl * t_
                    qdd = accl
                
                if t_b < t_ <= tf - t_b:
                    print("constant velocity phase")
                    q = (qf + q0 - V_max * tf)/2 + V_max*t_
                    qd = V_max
                    qdd = 0
                
                if tf - t_b < t_ <= tf:
                    print("deceleration phase")
                    q = qf - accl*tf**2/2 + accl*tf*t_ - accl*t_**2/2
                    qd = accl*tf - accl*t_
                    qdd = -accl
                
                q_arr = np.append(q_arr, q)
                qd_arr = np.append(qd_arr, qd)
                qdd_arr = np.append(qdd_arr, qdd)
            X[i, 0, :] = q_arr      # position
            X[i, 1, :] = qd_arr     # velocity
            X[i, 2, :] = qdd_arr    # acceleration

        return t, X

def main():
    ndof = 2
    # method = CubicPolynomial(ndof=ndof)
    # method = QuinticPolynomial(ndof=ndof)
    method = Trapezoidal(ndof=ndof)
    mode = "joint"

    # --------------------------------------------------------
    # Point-to-point multi-axis trajectory generator
    # --------------------------------------------------------

    # traj = MultiAxisTrajectoryGenerator(method=method,
    #                                     mode=mode,
    #                                     ndof=ndof)
    
    # traj.solve(q0=[-30, 30], qf=[60, 0], T=1)
    # traj.generate(nsteps=20)

    # --------------------------------------------------------
    # Via point multi-axis trajectory generator
    # --------------------------------------------------------

    traj = MultiSegmentTrajectoryGenerator(method=method,
                                           mode=mode,
                                           ndof=ndof,
                                            )
    via_points = [[-30, 30], [0, 45], [30, 15], [50, -30]]

    traj.solve(via_points, T=2)
    traj.generate(nsteps_per_segment=20)
    
    
    # plotter
    traj.plot()

if __name__ == "__main__":
    main()
